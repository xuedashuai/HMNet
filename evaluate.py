from __future__ import print_function
import torch
from model import HMNet
from utils import MyDataset, maskedTest
from torch.utils.data import DataLoader
import time
import numpy as np
from args import args

# Evaluation mode:
evalmode = 'write'  # rmse or rmse_all or ade or fde or all or write

# Initialize the network
net = HMNet(args)
net.load_state_dict(torch.load('trained_models/NGSIM_VA_UM.tar'))
net.training = False

if args['use_cuda']:
    net = net.cuda()
    
# Initialize the dataset
root = './data/'
testing_set = MyDataset(root + 'TestSet.mat')
testing_dataloader = DataLoader(testing_set, batch_size = args['batch_size'],
                                shuffle=True, num_workers=8, collate_fn=testing_set.collate_fn)

if args['use_feet'] == True:
    scale = 0.3048 # convert output from feet(ft) to meter(m)
else: scale = 1

# Initialize the caculation of RMSE/RMSE_ALL/ADE/FDE
lossVals   = torch.zeros(25).cuda()
lossVals_r = torch.zeros(25).cuda()
counts     = torch.zeros(25).cuda()

st_time = time.time()
for i, data in enumerate(testing_dataloader):
      
    loc_hist, loc_nbrs, mask, loc_fut, op_mask = data
    
    # Take the last frame of future track as destination
    dest = loc_fut[-1, :, :]# [L,N,H=2]
    
    # Initialize Variables
    if args['use_cuda']:
        loc_hist = loc_hist.cuda()
        loc_nbrs = loc_nbrs.cuda()
        mask = mask.cuda().bool()
        loc_fut =loc_fut.cuda()
        op_mask = op_mask.cuda()
        
    # Generate Social Latent Code    
    loc_enc, vel_enc, acc_enc = net.soc_encode(loc_hist,loc_nbrs,mask)
    
    if net.multi_modal:
        
        # Generate N guess in order to select the best guess  
        best_of_n = net.best_of_n
        all_l2_errors_dest = []
    	
        all_guesses = []
        for index in range(best_of_n):
    
            dest_recon = net(loc_hist, loc_nbrs, mask, None)
            dest_recon = dest_recon.detach().cpu().numpy()
            all_guesses.append(dest_recon)
            
            l2error_sample = np.linalg.norm(dest_recon - dest.numpy(), axis = 1)
            all_l2_errors_dest.append(l2error_sample)
            
        all_l2_errors_dest = np.array(all_l2_errors_dest)
        all_guesses = np.array(all_guesses)
    	# average error
        l2error_avg_dest = np.mean(all_l2_errors_dest)
    
    	# choosing the best guess
        indices = np.argmin(all_l2_errors_dest, axis = 0)
        best_guess_dest = all_guesses[indices,np.arange(loc_hist.shape[1]),:]
    
    	# taking the minimum error out of all guess
        l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))
    
    	# back to torch land
        best_guess_dest = torch.tensor(best_guess_dest).cuda()

    	# using the best guess for interpolation
        fut_pred = net.predict(loc_enc, vel_enc, acc_enc, best_guess_dest)
    else:
        fut_pred = net.predict(loc_enc, vel_enc, acc_enc)
    
    # Acquire the mean erro by maskedTest method
    l,l_r,c = maskedTest(fut_pred, loc_fut, op_mask)
    
    # Accumulate the erro
    lossVals   += l.detach()
    lossVals_r += l_r.detach()
    counts     += c.detach()
    
    if i%100 == 99:
        print("Testing progress(%):",format(i/(len(testing_set)/args['batch_size'])*100,'0.2f')) 

testing_time = time.time()-st_time
minutes = testing_time//60
seconds = testing_time%60

print('Testing Time:', minutes, 'min', seconds, 's')


if evalmode == 'rmse':
    res = torch.pow(lossVals / counts,0.5)*scale # Convert the output from feet to meter if use_feet is set
    print('RMSE:')
    print(res[4], res[9], res[14], res[19], res[24])   

elif evalmode == 'fde':
    res = (lossVals_r / counts) *scale
    print('FDE:')
    print(res[4], res[9], res[14], res[19], res[24])
    
elif evalmode == 'rmse_all':
    res = torch.pow(torch.sum(lossVals_r) / torch.sum(counts),0.5)*scale
    print('RMSE_ALL:')
    print(res)  

elif evalmode == 'ade':
    res1 = (torch.sum(lossVals_r[0:5]) / torch.sum(counts[0:5])) *scale
    res2 = (torch.sum(lossVals_r[0:10]) / torch.sum(counts[0:10])) *scale
    res3 = (torch.sum(lossVals_r[0:15]) / torch.sum(counts[0:15])) *scale
    res4 = (torch.sum(lossVals_r[0:20]) / torch.sum(counts[0:20])) *scale
    res5 = (torch.sum(lossVals_r[0:25]) / torch.sum(counts[0:25])) *scale
    print('ADE:')
    print(res1, res2, res3, res4, res5)

elif evalmode == 'all':
    # rmse
    res = torch.pow(lossVals / counts,0.5)*scale
    print('RMSE:')
    print(res[4], res[9], res[14], res[19], res[24])
    # fde
    res = (lossVals_r / counts) *scale
    print('FDE:')
    print(res[4], res[9], res[14], res[19], res[24])
    # rmse_all
    res = torch.pow(torch.sum(lossVals_r) / torch.sum(counts),0.5)*scale
    print('RMSE_ALL:')
    print(res)
    # ade
    res1 = (torch.sum(lossVals_r[0:5])  / torch.sum(counts[0:5])) *scale
    res2 = (torch.sum(lossVals_r[0:10]) / torch.sum(counts[0:10])) *scale
    res3 = (torch.sum(lossVals_r[0:15]) / torch.sum(counts[0:15])) *scale
    res4 = (torch.sum(lossVals_r[0:20]) / torch.sum(counts[0:20])) *scale
    res5 = (torch.sum(lossVals_r[0:25]) / torch.sum(counts[0:25])) *scale
    print('ADE:')
    print(res1, res2, res3, res4, res5)

elif evalmode == 'write':
    # openup test file
    f = open('evaluate.txt','r+')
    f.read()
    # Running time
    f.write('Time:\n')
    f.write('Testing Time:' + str(minutes) + 'min' + str(seconds) + 's'+'\n')
    # rmse
    res = (torch.pow(lossVals / counts,0.5)*scale).cpu().numpy()
    print('RMSE:')
    print(res)
    f.write('RMSE:\n')
    f.write(str(res[4])+' , '+str(res[9])+' , '+str(res[14])+' , '+str(res[19])+' , '+str(res[24])+'\n')
    # fde
    res = ((lossVals_r / counts) *scale).cpu().numpy()
    print('FDE:')
    print(res)
    f.write('FDE:\n')
    f.write(str(res[4])+' , '+str(res[9])+' , '+str(res[14])+' , '+str(res[19])+' , '+str(res[24])+'\n')
    # rmse_all
    res = (torch.pow(torch.sum(lossVals_r) / torch.sum(counts),0.5)*scale).cpu().numpy()
    print('RMSE_ALL:')
    print(res)
    f.write('RMSE_ALL:\n')
    f.write(str(res)+'\n')
    # ade
    res1 = ((torch.sum(lossVals_r[0:5])  / torch.sum(counts[0:5]))  *scale).cpu().numpy()
    res2 = ((torch.sum(lossVals_r[0:10]) / torch.sum(counts[0:10])) *scale).cpu().numpy()
    res3 = ((torch.sum(lossVals_r[0:15]) / torch.sum(counts[0:15])) *scale).cpu().numpy()
    res4 = ((torch.sum(lossVals_r[0:20]) / torch.sum(counts[0:20])) *scale).cpu().numpy()
    res5 = ((torch.sum(lossVals_r[0:25]) / torch.sum(counts[0:25])) *scale).cpu().numpy()
    print('ADE:')
    print(res1, res2, res3, res4, res5)
    f.write('ADE:\n')
    f.write(str(res1)+' , '+str(res2)+' , '+str(res3)+' , '+str(res4)+' , '+str(res5)+'\n')
    f.close()
