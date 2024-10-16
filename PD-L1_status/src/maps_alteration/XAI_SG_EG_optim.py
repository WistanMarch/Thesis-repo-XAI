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
    
    # Lists for SG & SG²
    all_attr_SG = []
    all_attr_SGSQ = []
    
    # For each sample
    for in_sample in samples:
        
        in_sample = in_sample.unsqueeze(0)
        
        # Empty list for storing all runs results
        raw_all_runs = []
        
        # For each run (random seed)
        for rseed in range (nbRuns):
            
            # print("\t\t\t EG Run", rseed+1)
            
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
        
        # Square of gradients for SmoothGrad Squared
        grads_SQ = (grads * grads)
        
        # Store sample attributions
        all_attr_SG.append(grads)
        all_attr_SGSQ.append(grads_SQ)
    
    # Divide by number of samples
    attr_SG = np.mean(all_attr_SG, axis=0)
    attr_SGSQ = np.mean(all_attr_SGSQ, axis=0)
    
    # Modify ID to integrate SG noise level
    filename_SG = filename[: filename.rfind('.')] + input_params['Noise'] + filename[filename.rfind('.') :]
    
    # Save attribution maps as .nii file
    save_file_SG = pathSave + 'Raw_attrs_SG+EG_' + filename_SG
    ni_img_SG = nib.Nifti1Image(attr_SG, affine=output_params['Affine'])
    nib.save(ni_img_SG, save_file_SG)
    save_file_SGSQ = pathSave + 'Raw_attrs_SGSQ+EG_' + filename_SG
    ni_img_SGSQ = nib.Nifti1Image(attr_SGSQ, affine=output_params['Affine'])
    nib.save(ni_img_SGSQ, save_file_SGSQ)
    
    # Compute & Save absolute if needed
    if input_params['Absolute']:
        pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(pathSave_abs, exist_ok=True)
        # Save absolute of SG map
        attr_SG_abs = np.abs(attr_SG)
        ni_img_SG_abs = nib.Nifti1Image(attr_SG_abs, affine=output_params['Affine'])
        nib.save(ni_img_SG_abs, pathSave_abs + 'Raw_attrs(absolute)_SG+EG_' + filename_SG)
        # Save SG² map without modification
        nib.save(ni_img_SGSQ, pathSave_abs + 'Raw_attrs(absolute)_SGSQ+EG_' + filename_SG)
