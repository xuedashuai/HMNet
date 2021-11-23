#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:28:26 2021

@author: seuiv
"""

## Network Arguments
args = {}
args['validation'] = False# Activate Validation part
args['writer_on'] = False # Activate Tensorboard summary writer
args['step_save'] = False # Save modal parameter during validation
args['use_feet'] = True # convert feet(ft) into meter(m) 
args['pretrain_epochs'] = 2 # MSEloss
args['train_epochs'] = 5 # NLL loss
args['batch_size']  = 128 

args['use_cuda'] = True

# Motion difference order:
#     0: None
#     1: Velocity (V)
#     2: Accerlation + Velocity (VA)
args['order'] = 2
args['best_of_n'] = 20

# Multi modality
args['multi_modal'] = False
args['encoder_size'] = 64
args['decoder_size'] = 128
args['hist_length'] = 16
args['fut_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['is_training'] = True

args['sampling_fre'] = 5

# Destination arguments
args['dest_dec_size'] = [128,64,32]
args['dest_latent_size'] = [8,50]
args['dest_enc_size'] = [8,16]
args['zdim'] = 16
args['fdim'] = 16
args['sigma'] = 1.3
