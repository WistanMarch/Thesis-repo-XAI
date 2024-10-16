# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:58:09 2022

@author: Wistan
"""


import os
import numpy as np
import nibabel as nib



#--------------------------------     NO PARAMETERS     --------------------------------#



#-------------------------------     BACKPROPAGATION     -------------------------------#



def Random(input_params, output_params, EG_params, GradCAM_params, IG_params):
    
    # Input data
    in_img = input_params['Numpy']
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    
    # Random 
    random_map = np.random.uniform(-1,1,(in_img.shape))
    
    # Save attribution map as .nii file
    save_file = pathSave + 'Raw_attrs_Random_' + output_params['ID']
    ni_img = nib.Nifti1Image(random_map, affine=output_params['Affine'])
    nib.save(ni_img, save_file)
    
    # Compute & Save absolute if needed
    if input_params['Absolute']:
        random_map_abs = np.abs(random_map)
        pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(pathSave_abs, exist_ok=True)
        ni_img_abs = nib.Nifti1Image(random_map_abs, affine=output_params['Affine'])
        nib.save(ni_img_abs, pathSave_abs + 'Raw_attrs(absolute)_Random_' + output_params['ID'])
