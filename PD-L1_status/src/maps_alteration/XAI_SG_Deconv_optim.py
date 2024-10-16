# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import numpy as np
import nibabel as nib
from captum.attr import Deconvolution



#-------------------------------     NO PARAMETERS     -------------------------------#



#-------------------------------     DECONVOLUTION     -------------------------------#



def Deconv(input_params, output_params, EG_params, GradCAM_params, IG_params): 
    
    # Set random seed
    np.random.seed(1)
    
    # Input data
    samples = input_params['Tensor']
    net = input_params['Network']
    label = int(input_params['Label'])
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    filename = output_params['ID']
    
    # Lists for SG & SG²
    all_attr_SG = []
    all_attr_SGSQ = []
    
    # For each sample
    for in_sample in samples:
        
        # Deconvolution init
        deconv = Deconvolution(net)
        
        # Inference
        in_sample = in_sample.unsqueeze(0)
        in_sample.requires_grad = True
        # in_img.requires_grad = True
        
        # Computes Deconvolution attribution scores
        grads = deconv.attribute(inputs=in_sample, target=label)
        # To numpy array
        grads = grads.squeeze().detach().cpu().numpy()
        
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
    save_file_SG = pathSave + 'Raw_attrs_SG+Deconv_' + filename_SG
    ni_img_SG = nib.Nifti1Image(attr_SG, affine=output_params['Affine'])
    nib.save(ni_img_SG, save_file_SG)
    save_file_SGSQ = pathSave + 'Raw_attrs_SGSQ+Deconv_' + filename_SG
    ni_img_SGSQ = nib.Nifti1Image(attr_SGSQ, affine=output_params['Affine'])
    nib.save(ni_img_SGSQ, save_file_SGSQ)
    
    # Compute & Save absolute if needed
    if input_params['Absolute']:
        pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(pathSave_abs, exist_ok=True)
        # Save absolute of SG map
        attr_SG_abs = np.abs(attr_SG)
        ni_img_SG_abs = nib.Nifti1Image(attr_SG_abs, affine=output_params['Affine'])
        nib.save(ni_img_SG_abs, pathSave_abs + 'Raw_attrs(absolute)_SG+Deconv_' + filename_SG)
        # Save SG² map without modification
        nib.save(ni_img_SGSQ, pathSave_abs + 'Raw_attrs(absolute)_SGSQ+Deconv_' + filename_SG)
