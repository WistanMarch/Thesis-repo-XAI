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



def save_nii(attr, extension, affine, output, trial_folder, filePrefix, filename):
    
    # Save folder
    save = output[:-1] + extension + output[-1] + trial_folder
    os.makedirs(save, exist_ok=True)
    
    # Save attribution maps as .nii file
    save_file = save + filePrefix + extension + '_' + filename
    ni_img = nib.Nifti1Image(attr, affine=affine)
    nib.save(ni_img, save_file)



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
    
    # Set random seed
    np.random.seed(1)
    
    # Input data
    samples = input_params['Numpy']
    net = input_params['Network']
    label = input_params['Label']
    # Output data
    pathSave = output_params['Savepath']
    filename = output_params['ID']
    filePrefix = input_params['FilePrefix']
    
    # Lists for SG
    all_attr_min = []
    all_attr_zero = []
    all_attr_max = []
    attr = ()
    
    # For each sample
    for in_sample in samples:
        
        # Construct the saliency object. This alone doesn't do anything.
        integrated_gradients = saliency.IntegratedGradients()
        
        # Arguments for call_model_function
        call_model_args = {'network': net, 'target':label}
        
        # Compute IG(Min) if necessary
        if ('Min' in filePrefix or 'Avg' in filePrefix):
            baseline_min = np.full(in_sample.shape, np.min(in_sample))
            grads_min = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_min, x_steps=200, batch_size=4)
            if (input_params['SG_type'] == 'SGSQ'): all_attr_min.append(grads_min * grads_min)
            else: all_attr_min.append(grads_min)
        
        # Compute IG(Zero) if necessary
        if ('Zero' in filePrefix or 'Avg' in filePrefix):
            baseline_zero = np.zeros_like(in_sample)
            grads_zero = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=4)
            if (input_params['SG_type'] == 'SGSQ'): all_attr_zero.append(grads_zero * grads_zero)
            else: all_attr_zero.append(grads_zero)
        
        # Compute IG(Max) if necessary
        if ('Max' in filePrefix or 'Avg' in filePrefix):
            baseline_max = np.full(in_sample.shape, np.max(in_sample))
            grads_max = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_max, x_steps=200, batch_size=4)
            if (input_params['SG_type'] == 'SGSQ'): all_attr_max.append(grads_max * grads_max)
            else: all_attr_max.append(grads_max)
    
    # Divide by number of samples
    if ('Min' in filePrefix or 'Avg' in filePrefix):
        attr_min = np.mean(all_attr_min, axis=0)
        attr = attr_min
    if ('Zero' in filePrefix or 'Avg' in filePrefix):
        attr_zero = np.mean(all_attr_zero, axis=0)
        attr = attr_zero
    if ('Max' in filePrefix or 'Avg' in filePrefix):
        attr_max = np.mean(all_attr_max, axis=0)
        attr = attr_max
    
    # Do averages if necessary
    if ('IG(Zero-Min)' in filePrefix):
        attr_min_stand = standardize_image(attr_min)
        attr_zero_stand = standardize_image(attr_zero)
        attr = np.mean([attr_zero_stand, attr_min_stand], axis=0)
    elif ('IG(Min-Max)' in filePrefix):
        attr_min_stand = standardize_image(attr_min)
        attr_max_stand = standardize_image(attr_max)
        attr = np.mean([attr_min_stand, attr_max_stand], axis=0)
    elif ('IG(Zero-Max)' in filePrefix):
        attr_zero_stand = standardize_image(attr_zero)
        attr_max_stand = standardize_image(attr_max)
        attr = np.mean([attr_zero_stand, attr_max_stand], axis=0)
    elif ('IG(Avg)' in filePrefix):
        attr_min_stand = standardize_image(attr_min)
        attr_zero_stand = standardize_image(attr_zero)
        attr_max_stand = standardize_image(attr_max)
        attr = np.mean([attr_min_stand, attr_zero_stand, attr_max_stand], axis=0)
    
    # If absolute map version
    if (input_params['Absolute']): attr = np.abs(attr)
    
    # Save attribution maps as .nii file
    save_file = pathSave + input_params['FilePrefix'] + '_' + filename
    ni_img = nib.Nifti1Image(attr, affine=output_params['Affine'])
    nib.save(ni_img, save_file)