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



def save_nii(attr_SG, attr_SGSQ, IG_method, affine, output, trial_folder, filename, do_absolute):
    
    # # Save folder
    # save = output[:-1] + extension + output[-1] + trial_folder
    # os.makedirs(save, exist_ok=True)
    
    # Save folder
    save = output + IG_method + '/' + trial_folder
    os.makedirs(save, exist_ok=True)
    
    # Save attribution maps as .nii file
    save_file_SG = save + 'Raw_attrs_SG+' + IG_method + '_' + filename
    ni_img_SG = nib.Nifti1Image(attr_SG, affine=affine)
    nib.save(ni_img_SG, save_file_SG)
    save_file_SGSQ = save + 'Raw_attrs_SGSQ+' + IG_method + '_' + filename
    ni_img_SGSQ = nib.Nifti1Image(attr_SGSQ, affine=affine)
    nib.save(ni_img_SGSQ, save_file_SGSQ)
    
    # Save absolute map if needed
    if (do_absolute):
        save_abs = save.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(save_abs, exist_ok=True)
        # Save absolute of SG map
        attr_SG_abs = np.abs(attr_SG)
        ni_img_SG_abs = nib.Nifti1Image(attr_SG_abs, affine=affine)
        nib.save(ni_img_SG_abs, save_abs + 'Raw_attrs(absolute)_SG+' + IG_method + '_' + filename)
        # Save SG² map without modification
        nib.save(ni_img_SGSQ, save_abs + 'Raw_attrs(absolute)_SGSQ+' + IG_method + '_' + filename)



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
    affine = output_params['Affine']
    output = IG_params['SaveBase']
    # pathSave = output_params['Savepath']
    # os.makedirs(pathSave, exist_ok=True)
    trial_folder = input_params['Trial']
    filename = output_params['ID']
    do_absolute = input_params['Absolute']
    IG_method = IG_params['Method']
    
    # Lists for SG & SG²
    all_attr_min_SG = []
    all_attr_zero_SG = []
    all_attr_max_SG = []
    all_attr_min_SGSQ = []
    all_attr_zero_SGSQ = []
    all_attr_max_SGSQ = []
    attr_min_SG = ()
    attr_min_SGSQ = ()
    attr_zero_SG = ()
    attr_zero_SGSQ = ()
    attr_max_SG = ()
    attr_max_SGSQ = ()
    attr_SG = ()
    attr_SGSQ = ()
    
    # For each sample
    for in_sample in samples:
        
        # Construct the saliency object. This alone doesn't do anything.
        integrated_gradients = saliency.IntegratedGradients()
    
        # Baselines : Min / Zero / Max values
        baseline_min = np.full(in_sample.shape, np.min(in_sample))
        baseline_zero = np.zeros_like(in_sample)
        baseline_max = np.full(in_sample.shape, np.max(in_sample))
        
        # Arguments for call_model_function
        call_model_args = {'network': net, 'target':label}
        
        # Compute IG(Min) if necessary
        if ('Min' in IG_method or 'Avg' in IG_method):
            baseline_min = np.full(in_sample.shape, np.min(in_sample))
            attr_min = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_min, x_steps=200, batch_size=4)
            all_attr_min_SGSQ.append(attr_min * attr_min)
            all_attr_min_SG.append(attr_min)
        
        # Compute IG(Zero) if necessary
        if ('Zero' in IG_method or 'Avg' in IG_method):
            baseline_zero = np.zeros_like(in_sample)
            attr_zero = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=4)
            all_attr_zero_SGSQ.append(attr_zero * attr_zero)
            all_attr_zero_SG.append(attr_zero)
        
        # Compute IG(Max) if necessary
        if ('Max' in IG_method or 'Avg' in IG_method):
            baseline_max = np.full(in_sample.shape, np.max(in_sample))
            attr_max = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_max, x_steps=200, batch_size=4)
            all_attr_max_SGSQ.append(attr_max * attr_max)
            all_attr_max_SG.append(attr_max)
        
        # # Compute Integrated Gradients for each baseline
        # attr_min = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_min, x_steps=200, batch_size=4)
        # attr_zero = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=4)
        # attr_max = integrated_gradients.GetMask(x_value=in_sample, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_max, x_steps=200, batch_size=4)
        
        # # Square of gradients for SmoothGrad Squared
        # attr_min_SQ = (attr_min * attr_min)
        # attr_zero_SQ = (attr_zero * attr_zero)
        # attr_max_SQ = (attr_max * attr_max)
        
        # # Store sample attributions
        # all_attr_min_SG.append(attr_min)
        # all_attr_zero_SG.append(attr_zero)
        # all_attr_max_SG.append(attr_max)
        # all_attr_min_SGSQ.append(attr_min_SQ)
        # all_attr_zero_SGSQ.append(attr_zero_SQ)
        # all_attr_max_SGSQ.append(attr_max_SQ)
    
    # Divide by number of samples
    if ('Min' in IG_method or 'Avg' in IG_method):
        attr_min_SG = np.mean(all_attr_min_SG, axis=0)
        attr_min_SGSQ = np.mean(all_attr_min_SGSQ, axis=0)
        # attr_SG = attr_min_SG
        # attr_SGSQ = attr_min_SGSQ
    if ('Zero' in IG_method or 'Avg' in IG_method):
        attr_zero_SG = np.mean(all_attr_zero_SG, axis=0)
        attr_zero_SGSQ = np.mean(all_attr_zero_SGSQ, axis=0)
        # attr_SG = attr_zero_SG
        # attr_SGSQ = attr_zero_SGSQ
    if ('Max' in IG_method or 'Avg' in IG_method):
        attr_max_SG = np.mean(all_attr_max_SG, axis=0)
        attr_max_SGSQ = np.mean(all_attr_max_SGSQ, axis=0)
        # attr_SG = attr_max_SG
        # attr_SGSQ = attr_max_SGSQ
    
    # # Divide by number of samples
    # attr_min_SG = np.mean(all_attr_min_SG, axis=0)
    # attr_zero_SG = np.mean(all_attr_zero_SG, axis=0)
    # attr_max_SG = np.mean(all_attr_max_SG, axis=0)
    # attr_min_SGSQ = np.mean(all_attr_min_SGSQ, axis=0)
    # attr_zero_SGSQ = np.mean(all_attr_zero_SGSQ, axis=0)
    # attr_max_SGSQ = np.mean(all_attr_max_SGSQ, axis=0)
    
    
    
    
    # Do averages if necessary
    if ('IG(Zero-Min)' in IG_method):
        attr_min_SG_stand = standardize_image(attr_min_SG)
        attr_zero_SG_stand = standardize_image(attr_zero_SG)
        attr_min_SGSQ_stand = standardize_image(attr_min_SGSQ)
        attr_zero_SGSQ_stand = standardize_image(attr_zero_SGSQ)
        attr_SG = np.mean([attr_zero_SG_stand, attr_min_SG_stand], axis=0)
        attr_SGSQ = np.mean([attr_zero_SGSQ_stand, attr_min_SGSQ_stand], axis=0)
    elif ('IG(Min-Max)' in IG_method):
        attr_min_SG_stand = standardize_image(attr_min_SG)
        attr_max_SG_stand = standardize_image(attr_max_SG)
        attr_min_SGSQ_stand = standardize_image(attr_min_SGSQ)
        attr_max_SGSQ_stand = standardize_image(attr_max_SGSQ)
        attr_SG = np.mean([attr_min_SG_stand, attr_max_SG_stand], axis=0)
        attr_SGSQ = np.mean([attr_min_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    elif ('IG(Zero-Max)' in IG_method):
        attr_zero_SG_stand = standardize_image(attr_zero_SG)
        attr_max_SG_stand = standardize_image(attr_max_SG)
        attr_zero_SGSQ_stand = standardize_image(attr_zero_SGSQ)
        attr_max_SGSQ_stand = standardize_image(attr_max_SGSQ)
        attr_SG = np.mean([attr_zero_SG_stand, attr_max_SG_stand], axis=0)
        attr_SGSQ = np.mean([attr_zero_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    elif ('IG(Avg)' in IG_method):
        attr_min_SG_stand = standardize_image(attr_min_SG)
        attr_zero_SG_stand = standardize_image(attr_zero_SG)
        attr_max_SG_stand = standardize_image(attr_max_SG)
        attr_min_SGSQ_stand = standardize_image(attr_min_SGSQ)
        attr_zero_SGSQ_stand = standardize_image(attr_zero_SGSQ)
        attr_max_SGSQ_stand = standardize_image(attr_max_SGSQ)
        attr_SG = np.mean([attr_min_SG_stand, attr_zero_SG_stand, attr_max_SG_stand], axis=0)
        attr_SGSQ = np.mean([attr_min_SGSQ_stand, attr_zero_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    
    
    
    
    # # Standardize maps for averages
    # attr_min_SG_stand = standardize_image(attr_min_SG)
    # attr_zero_SG_stand = standardize_image(attr_zero_SG)
    # attr_max_SG_stand = standardize_image(attr_max_SG)
    # attr_min_SGSQ_stand = standardize_image(attr_min_SGSQ)
    # attr_zero_SGSQ_stand = standardize_image(attr_zero_SGSQ)
    # attr_max_SGSQ_stand = standardize_image(attr_max_SGSQ)
    
    # # Two-by-two combined attribution maps
    # attr_zero_min_SG = np.mean([attr_zero_SG_stand, attr_min_SG_stand], axis=0)
    # attr_zero_max_SG = np.mean([attr_zero_SG_stand, attr_max_SG_stand], axis=0)
    # attr_min_max_SG = np.mean([attr_min_SG_stand, attr_max_SG_stand], axis=0)
    # attr_zero_min_SGSQ = np.mean([attr_zero_SGSQ_stand, attr_min_SGSQ_stand], axis=0)
    # attr_zero_max_SGSQ = np.mean([attr_zero_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    # attr_min_max_SGSQ = np.mean([attr_min_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    # # Create a combined attribution map (average of all)
    # attr_mean_SG = np.mean([attr_min_SG_stand, attr_zero_SG_stand, attr_max_SG_stand], axis=0)
    # attr_mean_SGSQ = np.mean([attr_min_SGSQ_stand, attr_zero_SGSQ_stand, attr_max_SGSQ_stand], axis=0)
    
    # Modify ID to integrate SG noise level
    filename_SG = filename[: filename.rfind('.')] + input_params['Noise'] + filename[filename.rfind('.') :]
    
    # Save all computed maps
    if len(attr_SG) != 0:
        save_nii(attr_SG, attr_SGSQ, IG_method, affine, output, trial_folder, filename_SG, do_absolute)
    if len(attr_min_SG) != 0:
        save_nii(attr_min_SG, attr_min_SGSQ, 'IG(Min)', affine, output, trial_folder, filename_SG, do_absolute)
    if len(attr_zero_SG) != 0:
        save_nii(attr_zero_SG, attr_zero_SGSQ, 'IG(Zero)', affine, output, trial_folder, filename_SG, do_absolute)
    if len(attr_max_SG) != 0:
        save_nii(attr_max_SG, attr_max_SGSQ, 'IG(Max)', affine, output, trial_folder, filename_SG, do_absolute)
    
    
    
    
    # # Save all attribution maps (original & absolute)
    # save_nii(attr_min_SG, attr_min_SGSQ, '(Min)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_zero_SG, attr_zero_SGSQ, '(Zero)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_max_SG, attr_max_SGSQ, '(Max)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_zero_min_SG, attr_zero_min_SGSQ, '(Zero-Min)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_zero_max_SG, attr_zero_max_SGSQ, '(Zero-Max)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_min_max_SG, attr_min_max_SGSQ, '(Min-Max)', affine, output, trial_folder, filename_SG, do_absolute)
    # save_nii(attr_mean_SG, attr_mean_SGSQ, '(Avg)', affine, output, trial_folder, filename_SG, do_absolute)
