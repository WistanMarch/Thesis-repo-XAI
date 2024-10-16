# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import nibabel as nib
import saliency.core as saliency
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
from skimage.transform import resize



#------------------------------------     PARAMETERS     ------------------------------------#


# Expected keys for GradCAM
expected_keys = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#------------------------------------     Hooks Functions     ------------------------------------#


conv_layer_outputs = {} 
def conv_layer_forward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, -1).detach().cpu().numpy()
def conv_layer_backward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, -1).detach().cpu().numpy()


#------------------------------------     Custom GradCAM     ------------------------------------#


# Function was modified to accept 3D input
def compute_GradCAM(x_value,
            call_model_function,
            call_model_args=None,
            should_resize=True,
            three_dims=True):
    
    x_value_batched = np.expand_dims(x_value, axis=0)
    data = call_model_function( x_value_batched,
                                call_model_args=call_model_args,
                                expected_keys=expected_keys)
    
    weights = np.mean(data[CONVOLUTION_OUTPUT_GRADIENTS][0], axis=(0, 1, 2))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER_VALUES][0].shape[0:3],
                        dtype=np.float32)

    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER_VALUES][0][:, :, :, i]

    # pass through relu
    grad_cam = np.maximum(grad_cam, 0)

    # resize heatmap to be the same size as the input
    if should_resize:
      if np.max(grad_cam) > 0:
        grad_cam = grad_cam / np.max(grad_cam)
      grad_cam = resize(grad_cam, x_value.shape[:3])

    return grad_cam


#------------------------------------     call_model_function     ------------------------------------#


def call_model_function(images,
                        call_model_args=None,
                        expected_keys=None):
    
    handle_forward = call_model_args['layer'].register_forward_hook(conv_layer_forward)
    handle_backward = call_model_args['layer'].register_full_backward_hook(conv_layer_backward)

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = call_model_args['target']
    output = call_model_args['network'](images)
    
    target = output[:,target_class_idx]
    call_model_args['network'].zero_grad()
    _ = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
    
    handle_forward.remove()
    handle_backward.remove()
    
    return conv_layer_outputs


#------------------------------------     GRADCAM     ------------------------------------#



# def GradCAM(output, do_absolute):
def GradCAM(input_params, output_params, EG_params, GradCAM_params, IG_params):
    
    # Input data
    in_img = input_params['Numpy']
    # Output path
    pathSave = output_params['Savepath']
    os.makedirs(pathSave, exist_ok=True)
    
    call_model_args = {'network': input_params['Network'], 'layer': GradCAM_params['Layer'], 'target':input_params['Label']}
    
    # Compute the Grad-CAM mask
    grad_cam_mask = compute_GradCAM(x_value=in_img, call_model_function=call_model_function, call_model_args=call_model_args, should_resize=True, three_dims=False)
    
    # Save attribution map as .nii file
    save_file = pathSave + 'Raw_attrs_GradCAM_' + output_params['ID']
    ni_img = nib.Nifti1Image(grad_cam_mask, affine=output_params['Affine'])
    nib.save(ni_img, save_file)
    
    # Save absolute (same map)
    pathSave_abs = pathSave.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
    os.makedirs(pathSave_abs, exist_ok=True)
    nib.save(ni_img, pathSave_abs + 'Raw_attrs(absolute)_GradCAM_' + output_params['ID'])
