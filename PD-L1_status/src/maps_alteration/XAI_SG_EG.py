# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import shap
import nibabel as nib
from monai.data import ImageDataset, DataLoader



#------------------------------------     PARAMETERS     ------------------------------------#


# Number of EG runs & Train batches size
nbRuns = 1
batch_size = 1
nb_samples = 4


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#-------------------------------     EXPECTED GRADIENTS     ------------------------------#



def EG(input_params, output_params, EG_params, GradCAM_params, IG_params):
    
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
        
        in_sample = in_sample.unsqueeze(0)
        
        # Empty list for storing all runs results
        raw_all_runs = []
        
        # For each run (random seed)
        for rseed in range (nbRuns):
            
            # Prepare the training dataset (each run transforms are different)
            train_ds = ImageDataset(image_files=EG_params['Imgs'], labels=EG_params['Labels'], transform=EG_params['Transforms'])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            # One run attributions
            raw_all_batches = []
            
            # For each element in test loader
            for train_step, train_data in enumerate(train_loader):
            
                # Load original input
                batch_tensor = train_data[0].to(device)
                
                # GradientExplainer
                e = shap.GradientExplainer(net, batch_tensor)
                
                # Compute SHAP values for ID (4 samples)
                shap_values, _ = e.shap_values(in_sample, nsamples=nb_samples, ranked_outputs=1, rseed=rseed)
                
                # Save as part of one run
                for _ in range(len(batch_tensor)): raw_all_batches.append(shap_values[0].squeeze())
            
            # Avg map for one run
            raw_one_run = np.mean(raw_all_batches, axis=0)
            
            # Append run map to list
            raw_all_runs.append(raw_one_run)
    
        # Mean of all runs
        grads = np.mean(raw_all_runs, axis=0)
        
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