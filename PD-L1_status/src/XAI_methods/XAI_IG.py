# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import nibabel as nib
import saliency.core as saliency



#------------------------------------     PARAMETERS     ------------------------------------#


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#-------------------------------     Standardization     -------------------------------#



def standardize_image(im):
    extreme_value = np.max(np.abs(im))
    return im / extreme_value



#-------------------------------     Save Function     -------------------------------#



def save_nii(attr, extension, affine, output, trial_folder, filename, do_absolute):
    
    # Save attribution map as .nii
    save = output[:-1] + extension + output[-1] + trial_folder
    os.makedirs(save, exist_ok=True)
    save_file = save + 'Raw_attrs_IG' + extension + '_' + filename
    ni_img = nib.Nifti1Image(attr, affine=affine)
    nib.save(ni_img, save_file)
    
    # Save absolute map if needed
    if (do_absolute):
        attr_abs = np.abs(attr)
        save_abs = save.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(save_abs, exist_ok=True)
        ni_img_abs = nib.Nifti1Image(attr_abs, affine=affine)
        nib.save(ni_img_abs, save_abs + 'Raw_attrs(absolute)_IG' + extension + '_' + filename)



#-------------------------------     call_model_function     -------------------------------#



def call_model_function(images,
                        call_model_args=None,
                        expected_keys=None):
    
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    
    target_class_idx = call_model_args['target']
    output = call_model_args['network'](images)
    
    target = output[:,target_class_idx]
    call_model_args['network'].zero_grad()
    
    grads = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target))[0].squeeze()
    gradients = grads.cpu().detach().numpy()
    
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



#-------------------------------     INTEGRATED GRADIENTS     -------------------------------#



def IG(input_params, output_params, EG_params, GradCAM_params, IG_params):
    
    
    # Input data
    in_img = input_params['Numpy']
    net = input_params['Network']
    label = input_params['Label']
    
    # Output data
    affine = output_params['Affine']
    output = IG_params['SaveBase'] + 'IG/'
    trial_folder = input_params['Trial']
    filename = output_params['ID']
    do_absolute = input_params['Absolute']
    
    # Construct the saliency object. This alone doesn't do anything.
    integrated_gradients = saliency.IntegratedGradients()

    # Baselines : Min / Zero / Max values
    baseline_min = np.full(in_img.shape, np.min(in_img))
    baseline_zero = np.zeros_like(in_img)
    baseline_max = np.full(in_img.shape, np.max(in_img))
    
    # Arguments for call_model_function
    call_model_args = {'network': net, 'target':label}
    
    # Compute Integrated Gradients for each baseline
    attr_min = integrated_gradients.GetMask(x_value=in_img, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_min, x_steps=200, batch_size=4)
    attr_zero = integrated_gradients.GetMask(x_value=in_img, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=4)
    attr_max = integrated_gradients.GetMask(x_value=in_img, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_max, x_steps=200, batch_size=4)
    
    # Standardize maps for averages
    attr_min_stand = standardize_image(attr_min)
    attr_zero_stand = standardize_image(attr_zero)
    attr_max_stand = standardize_image(attr_max)
    
    # Two-by-two combined attribution maps
    attr_zero_min = np.mean([attr_zero_stand, attr_min_stand], axis=0)
    attr_zero_max = np.mean([attr_zero_stand, attr_max_stand], axis=0)
    attr_min_max = np.mean([attr_min_stand, attr_max_stand], axis=0)
    # Create a combined attribution map (average of all)
    attr_mean = np.mean([attr_min_stand, attr_zero_stand, attr_max_stand], axis=0)
    
    # Save all attribution maps (original & absolute)
    save_nii(attr_min, '(Min)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_zero, '(Zero)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_max, '(Max)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_zero_min, '(Zero-Min)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_zero_max, '(Zero-Max)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_min_max, '(Min-Max)', affine, output, trial_folder, filename, do_absolute)
    save_nii(attr_mean, '(Avg)', affine, output, trial_folder, filename, do_absolute)
