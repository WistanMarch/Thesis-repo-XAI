# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import saliency.core as saliency
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
import torch.nn as nn
from utils.XAI_utils import XAI_dataset
import nibabel as nib
from skimage.transform import resize



#------------------------------------     PARAMETERS     ------------------------------------#


# Base paths
loading = './'


# List of folds
trials = [
            0,
            1,
            2,
            3,
            4,
          ]


# Fixed Parameters (since training is complete)
params = {
            'net_id': 11,
            'feature': 'CPS',
            'cutoff': 1,
            'batch': 1,
            'size': 192,
            'offsetx': 0,
            'offsety': 0,
            'offsetz': 0,
         }


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


#------------------------------------     GradCAM     ------------------------------------#


# Function was modified to accept 3D input
def GradCAM(x_value,
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
    
    handle_forward = conv_layer.register_forward_hook(conv_layer_forward)
    handle_backward = conv_layer.register_full_backward_hook(conv_layer_backward)

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = call_model_args['target']
    output = net(images)
    
    target = output[:,target_class_idx]
    net.zero_grad()
    _ = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
    
    handle_forward.remove()
    handle_backward.remove()
    
    return conv_layer_outputs


#------------------------------------     MAIN     ------------------------------------#


# Defining main function
def main(layer_name): 
    
    print('\t Layer', layer_name)

    # For each element in loader
    for step, data in enumerate(loader):
        
        # Load original input
        in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
        in_numpy = in_tensor.squeeze().cpu().numpy()
        affine = data[0].meta['affine'].squeeze()
        
        call_model_args = {'target': label}
        
        # Compute the Grad-CAM mask
        grad_cam_mask = GradCAM(x_value=in_numpy, call_model_function=call_model_function, call_model_args=call_model_args, should_resize=True, three_dims=False)
        
        # Save attribution map as .nii file
        ni_img = nib.Nifti1Image(grad_cam_mask, affine=affine)
        nib.save(ni_img, pathResults + 'Raw_attrs_GradCam_' + ids[step] + '_' + str(count) + '-' + layer_name + '.nii')



#------------------------------------     CALL MAIN     ------------------------------------#


if __name__=="__main__":
    
    # For each trial (=fold)
    for trial in trials:
        
        print('Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, net = XAI_dataset(params, trial, device, loading)
        
        # Path to save GradCAM attributions
        pathResults = loading + 'Results/GradCAM/' + trial_folder + '/Raw_attrs/'
        os.makedirs(pathResults, exist_ok=True)
        
        # Get list of conv layers
        conv_layers = []
        conv_names = []
        for i, d in enumerate(net.named_modules()):
            if ('project' not in d[0]) and (isinstance(d[1], nn.Conv3d)):
                conv_names.append(d[0])
                conv_layers.append(d[1])
    
        # Compute GradCAM maps
        for count, conv_layer in enumerate(conv_layers):
            main(conv_names[count])