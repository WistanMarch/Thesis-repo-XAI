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
    
    
    # Input data
    in_img = input_params['Tensor']
    net = input_params['Network']
    label = int(input_params['Label'])
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    
    # Deconvolution init
    deconv = Deconvolution(net)
    
    # Inference
    in_img.requires_grad = True
    
    # Computes Deconvolution attribution scores
    grads = deconv.attribute(inputs=in_img, target=label)
    # To numpy array
    grads = grads.squeeze().detach().cpu().numpy()
    
    # Save attribution map as .nii file
    save_file = pathSave + 'Raw_attrs_Deconv_' + output_params['ID']
    ni_img = nib.Nifti1Image(grads, affine=output_params['Affine'])
    nib.save(ni_img, save_file)
    
    # Compute & Save absolute if needed
    if input_params['Absolute']:
        grads_abs = np.abs(grads)
        pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(pathSave_abs, exist_ok=True)
        ni_img_abs = nib.Nifti1Image(grads_abs, affine=output_params['Affine'])
        nib.save(ni_img_abs, pathSave_abs + 'Raw_attrs(absolute)_Deconv_' + output_params['ID'])
