from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation, MLP, histDifference
from args import args

    
class HMNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(HMNet, self).__init__()
        
        self.training = True
        self.multi_modal = args['multi_modal']
        self.best_of_n = args['best_of_n'] 
        self.order = args['order']
        self.sampling_fre = args['sampling_fre']
        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.hist_length = args['hist_length']
        self.fut_length = args['fut_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        
        self.dest_enc_size = args['dest_enc_size']
        self.dest_latent_size = args['dest_latent_size']
        self.dest_dec_size = args['dest_dec_size']

        self.fdim = args['fdim']
        self.zdim = args['zdim']
        self.sigma = args['sigma']
        
        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))
        
        # Loc
        self.loc_emb = torch.nn.Linear(2,self.input_embedding_size)
        self.loc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)
        self.loc_dyn = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
        self.loc_op = torch.nn.Linear(self.decoder_size,5)
        
        if self.multi_modal:
            if self.order == 0:
                self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.fdim, self.decoder_size)
            else:
                self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.decoder_size + self.fdim, self.decoder_size)
        else:
            if self.order == 0:
                self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)
            else:
                self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.decoder_size, self.decoder_size)
            
        # Vel
        if not self.order == 0:
            
            self.vel_emb = torch.nn.Linear(2,self.input_embedding_size)
            self.vel_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)
            self.vel_dyn = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
            self.vel_op = torch.nn.Linear(self.decoder_size,5)
            
            if self.order == 1:
                
                self.trans = torch.nn.Linear(2 * self.encoder_size, self.encoder_size)
                self.vel_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)
                
            if self.order == 2:
                
                self.trans = torch.nn.Linear(3 * self.encoder_size, self.encoder_size)
                self.vel_dec = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.decoder_size, self.decoder_size)
                
                # Acc
                self.acc_emb = torch.nn.Linear(2,self.input_embedding_size)
                self.acc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)
                self.acc_dyn = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)
                self.acc_dec = torch.nn.LSTM(self.dyn_embedding_size + self.soc_embedding_size, self.decoder_size)
                self.acc_op = torch.nn.Linear(self.decoder_size,5)
        
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Endpoint
        self.dest_enc = MLP(input_dim = 2, output_dim = self.fdim, hidden_size=self.dest_enc_size)
        self.latent_enc = MLP(input_dim = self.fdim + self.dyn_embedding_size + self.soc_embedding_size, output_dim = 2*self.zdim, hidden_size=self.dest_latent_size)
        self.dest_dec = MLP(input_dim = self.dyn_embedding_size + self.soc_embedding_size + self.zdim, output_dim = 2, hidden_size=self.dest_dec_size)
            
    def encode(self, emb, lstm, dyn, ip, nbrs):
        ## Forward pass hist:
        _,(hist_enc,_) = lstm(self.leaky_relu(emb(ip)))
        hist_enc = self.leaky_relu(dyn(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc,_) = lstm(self.leaky_relu(emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        
        return hist_enc, nbrs_enc
    
    def soc_pooling(self, masks, ip):
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float().cuda()
        soc_enc = soc_enc.masked_scatter_(masks, ip)
        soc_enc = soc_enc.permute(0,3,2,1)

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)
        
        return soc_enc
    
    def soc_encode(self,loc_hist,loc_nbrs,masks):
        vel_enc = None
        acc_enc = None
        
        vel_hist, vel_nbrs, acc_hist, acc_nbrs = histDifference(loc_hist, loc_nbrs, length = self.hist_length, delta = 1.0/self.sampling_fre, order = self.order)

        hist_enc, nbrs_enc = self.encode(self.loc_emb, self.loc_lstm, self.loc_dyn, loc_hist, loc_nbrs)
        
        if not self.order == 0:
            vel_enc, nbrs_vel_enc = self.encode(self.vel_emb, self.vel_lstm, self.vel_dyn, vel_hist, vel_nbrs)
            
            if self.order == 1:
                all_enc = torch.cat((nbrs_enc, nbrs_vel_enc), axis = 1)
                
            if self.order == 2:
                acc_enc, nbrs_acc_enc = self.encode(self.acc_emb, self.acc_lstm, self.acc_dyn, acc_hist, acc_nbrs)
                all_enc = torch.cat((nbrs_enc, nbrs_vel_enc, nbrs_acc_enc), axis = 1)
                
            all_enc = self.trans(all_enc)
            
        else:
            all_enc = nbrs_enc
        
        soc_enc = self.soc_pooling(masks, all_enc)

        ## Concatenate encodings:        
        loc_enc = torch.cat((soc_enc,hist_enc),1)
        
        if not self.order == 0:
            vel_enc = torch.cat((soc_enc,vel_enc), axis = 1)
            
            if self.order == 2:
                acc_enc = torch.cat((soc_enc,acc_enc), axis = 1)
        
        return loc_enc, vel_enc, acc_enc
    
    ## Forward Pass
    def forward(self,loc_hist,loc_nbrs,masks,dest):
        loc_enc, vel_enc, acc_enc = self.soc_encode(loc_hist,loc_nbrs,masks)
        
        if self.multi_modal:
            if self.training:
                # CVAE code
                dest_features = self.dest_enc(dest)
                features = torch.cat((loc_enc, dest_features), dim = 1)
                latent =  self.latent_enc(features)
    
                mu = latent[:, 0:self.zdim] # 2-d array
                logvar = latent[:, self.zdim:] # 2-d array
    
                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_()
                eps = eps.cuda()
                z = eps.mul(var).add_(mu)
            
            else:
                z = torch.Tensor(loc_hist.shape[1], self.zdim)  
                z.normal_(0, self.sigma)
            
            z = z.float().cuda()
            decoder_input = torch.cat((loc_enc, z), dim = 1)
            generated_dest = self.dest_dec(decoder_input)
            
            if self.training:
                
                loc_pred, vel_pred, acc_pred = self.predict(loc_enc, vel_enc, acc_enc, generated_dest)    
                return loc_pred, vel_pred, acc_pred, generated_dest, mu, logvar
            else:
                return generated_dest
        else:
            loc_pred, vel_pred, acc_pred = self.predict(loc_enc, vel_enc, acc_enc)
            
            return loc_pred, vel_pred, acc_pred

    def predict(self,loc_enc, vel_enc, acc_enc, generated_dest = None):
        
        if not self.order == 0:
            # Vel
            vel_enc = vel_enc.repeat(self.fut_length, 1, 1)
            
            if self.order == 2:
                # ACC
                acc_enc = acc_enc.repeat(self.fut_length, 1, 1)
                h_acc, _ = self.acc_dec(acc_enc)
                
                vel_enc = torch.cat((vel_enc, h_acc), axis = 2)
            
            h_vel, _ = self.vel_dec(vel_enc)
        
        
        if self.training:
            vel_pred = None
            acc_pred = None
            
            if self.multi_modal:
                generated_dest_features = self.dest_enc(generated_dest)
                loc_enc = torch.cat((loc_enc, generated_dest_features), axis = 1)
                
            loc_enc = loc_enc.repeat(self.fut_length, 1, 1)
            
            if not self.order == 0:
                vel_pred = self.vel_op(h_vel)
                vel_pred = outputActivation(vel_pred)
            
                if self.order == 2:
                    acc_pred = self.acc_op(h_acc)
                    acc_pred = outputActivation(acc_pred)
                    
                loc_enc = torch.cat((loc_enc, h_vel), axis = 2)
            
            h_dec, _ = self.loc_dec(loc_enc)
            h_dec = h_dec.permute(1, 0, 2)
            fut_pred = self.loc_op(h_dec)
            fut_pred = fut_pred.permute(1, 0, 2)
            fut_pred = outputActivation(fut_pred)
            
            return fut_pred, vel_pred, acc_pred
        
        else:
            
            if self.multi_modal:
                generated_dest_features = self.dest_enc(generated_dest)
                loc_enc = torch.cat((loc_enc, generated_dest_features), axis = 1)
            
            loc_enc = loc_enc.repeat(self.fut_length, 1, 1)
            
            if not self.order == 0:
                loc_enc = torch.cat((loc_enc, h_vel), axis = 2)
            
            h_dec, _ = self.loc_dec(loc_enc)
            h_dec = h_dec.permute(1, 0, 2)
            fut_pred = self.loc_op(h_dec)
            fut_pred = fut_pred.permute(1, 0, 2)
            fut_pred = outputActivation(fut_pred)
            
            return fut_pred





