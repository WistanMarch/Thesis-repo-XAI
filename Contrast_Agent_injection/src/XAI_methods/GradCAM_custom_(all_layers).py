# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import pickle
import saliency.core as saliency
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
from skimage.transform import resize
import torch.nn as nn



#------------------------------------     PARAMETERS     ------------------------------------#


# Base paths
pathRoot = './'
# pathRoot = '/home/psoto/Data/WORK/LaTIM/CODE/LaTIM/Interpretability-Radiomics/Part_1_Contrast_Agent_Classification/'


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Expected keys for GradCAM
expected_keys = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]


#------------------------------------     Hooks Functions     ------------------------------------#


conv_layer_outputs = {} 
def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()
    

#------------------------------------     Custom GradCAM     ------------------------------------#


def GradCAM(x_value,
            call_model_function,
            call_model_args=None,
            should_resize=True,
            three_dims=True):

    x_value_batched = np.expand_dims(x_value, axis=0)
    data, output = call_model_function(x_value_batched,
                                     call_model_args=call_model_args,
                                     expected_keys=expected_keys)
    
    weights = np.mean(data[CONVOLUTION_OUTPUT_GRADIENTS][0], axis=(0, 1))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER_VALUES][0].shape[0:2], dtype=np.float32)
    
    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER_VALUES][0][:, :, i]
    
    # Use max or min depending on pred
    pred = torch.sigmoid(output).detach().cpu().numpy()
    if (pred <= 0.5):
        grad_cam = np.abs(np.minimum(grad_cam, 0))
    else:
        grad_cam = np.maximum(grad_cam, 0)
    
    # resize heatmap to be the same size as the input
    if should_resize:
        if np.max(grad_cam) > 0:
            grad_cam = grad_cam / np.max(grad_cam)
        grad_cam = resize(grad_cam, x_value.shape[:2])
    
    # convert grayscale to 3-D
    if three_dims:
        grad_cam = np.expand_dims(grad_cam, axis=2)
        grad_cam = np.tile(grad_cam, [1, 1, 3])
    
    return grad_cam


#------------------------------------     call_model_function     ------------------------------------#


def call_model_function(images,
                        call_model_args=None,
                        expected_keys=None):
    
    handle_forward = conv_layer.register_forward_hook(conv_layer_forward)
    handle_backward = conv_layer.register_full_backward_hook(conv_layer_backward)

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    output = model(images)
    
    target = output[:,target_class_idx]
    model.zero_grad()
    _= torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
    
    handle_forward.remove()
    handle_backward.remove()
    
    return conv_layer_outputs, output


#------------------------------------     MAIN     ------------------------------------#


# Defining main function
def main(count, layer_name): 
    
    print('\t Layer', layer_name)

    # For each input image we display and save the results
    for sliceIdx in range (len(test_images)):
        # Load input image and raw attributions for given slice and method
        im_tensor = test_images[sliceIdx].unsqueeze(0)
        im_numpy = im_tensor.squeeze().cpu().numpy()
        
        # Compute GradCAM
        grad_cam_mask = GradCAM(x_value=im_numpy, call_model_function=call_model_function, call_model_args = None, should_resize=True, three_dims=False)
        
        # Raw attributions save
        os.makedirs(pathResults, exist_ok=True)
        np.save(pathResults + 'Raw_attr_GradCam_' + str(sliceIdx) + '_' + str(count) + '-' + layer_name + '.npy', grad_cam_mask)



#------------------------------------     CALL MAIN     ------------------------------------#


if __name__=="__main__":
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load images
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/GradCAM/' + arch + '/Raw_attrs/'
        os.makedirs(pathResults, exist_ok=True)
        
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
    
        # Get list of conv layers
        conv_layers = []
        conv_names = []
        for i, d in enumerate(model.named_modules()):
            if (i != 0) and (isinstance(d[1], nn.Conv2d)):
                if (arch == 'vgg19') or ('conv' in d[0]):
                    conv_names.append(d[0])
                    conv_layers.append(d[1])
    
        # Compute GradCAM maps
        for count, conv_layer in enumerate(conv_layers):
            main(count, conv_names[count])