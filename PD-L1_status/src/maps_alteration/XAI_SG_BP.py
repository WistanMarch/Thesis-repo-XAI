# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import nibabel as nib



#--------------------------------     NO PARAMETERS     --------------------------------#



#-------------------------------     BACKPROPAGATION     -------------------------------#



def BP(input_params, output_params, EG_params, GradCAM_params, IG_params):
    
    # Set random seed
    np.random.seed(1)
    
    # Input data
    samples = input_params['Tensor']
    net = input_params['Network']
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    filename = output_params['ID']
    
    # List for SG attributions
    all_attrs = []
    
    # For each sample
    for in_sample in samples:
    
        # Inference
        in_sample = in_sample.unsqueeze(0)
        in_sample.requires_grad = True
        outputs = net(in_sample)
        # Select highest output class score
        target = outputs[:, outputs.argmax()]
        
        # Get gradients
        net.zero_grad()
        grads = torch.autograd.grad(target, in_sample, grad_outputs=torch.ones_like(target))[0].squeeze()
        grads = grads.cpu().detach().numpy()
        
        # Append to grads of samples
        if (input_params['SG_type'] == 'SGSQ'): all_attrs.append(grads * grads)
        else: all_attrs.append(grads)
    
    # Divide by number of samples
    attr = np.mean(all_attrs, axis=0)
    
    # If absolute map version
    if (input_params['Absolute']): attr = np.abs(attr)
    
    # Save attribution maps as .nii file
    save_file = pathSave + input_params['FilePrefix'] + '_' + filename
    ni_img = nib.Nifti1Image(attr, affine=output_params['Affine'])
    nib.save(ni_img, save_file)