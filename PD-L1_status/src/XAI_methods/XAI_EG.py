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
    
    # Input data
    in_img = input_params['Tensor']
    net = input_params['Network']
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    
    # Empty list for storing all runs results
    raw_all_runs = []
    
    # For each run (random seed)
    for rseed in range (nbRuns):
        
        print("\t\t\t EG Run", rseed+1)
        
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
            shap_values, _ = e.shap_values(in_img, nsamples=nb_samples, ranked_outputs=1, rseed=rseed)
            
            # Save as part of one run
            for _ in range(len(batch_tensor)): raw_all_batches.append(shap_values[0].squeeze())
        
        # Avg map for one run
        raw_one_run = np.mean(raw_all_batches, axis=0)
        
        # Append run map to list
        raw_all_runs.append(raw_one_run)

    # Mean of all runs
    grads = np.mean(raw_all_runs, axis=0)
    
    # Save attribution map as .nii file
    save_file = pathSave + 'Raw_attrs_EG_' + output_params['ID']
    ni_img = nib.Nifti1Image(grads, affine=output_params['Affine'])
    nib.save(ni_img, save_file)
    
    # Compute & Save absolute if needed
    if input_params['Absolute']:
        grads_abs = np.abs(grads)
        pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(pathSave_abs, exist_ok=True)
        ni_img_abs = nib.Nifti1Image(grads_abs, affine=output_params['Affine'])
        nib.save(ni_img_abs, pathSave_abs + 'Raw_attrs(absolute)_EG_' + output_params['ID'])
