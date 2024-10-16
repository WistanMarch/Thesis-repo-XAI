# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:42:25 2021

@author: Wistan
"""


import os
import itk
import torch
import numpy as np
import pickle

from skimage.transform import resize

import shap
import saliency.core as saliency
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS



#-----------------------------     Parameters     -----------------------------#



# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathImgReg = pathResults + 'ImgReg/'
pathTransformed = pathImgReg + 'Transformed/'



# List of trained networks
networks = {
            "resnet" : "layer4.2.conv1",
            # "Xception" : "conv4.conv1",
            }


# All methods for files loading
INTERPRET_METHODS = [
                        'BP',
                        'IG(0)',
                        'EG',
                        'GradCAM',
                     ]


# Expected keys for GradCAM
expected_keys_GradCAM = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]


# Reference image index (from Image_Registration.py)
reference_idx = 32


# Number of runs for EG
nbRunsEG = 1


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test images file name
test_images_filename = 'test_slices_260'



#------------------------------------------------------------------------------#

#---------------------------     Utilitaries     ------------------------------#



conv_layer_outputs = {} 
def conv_layer_forward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()
def conv_layer_backward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()
    


def call_model_function(images, call_model_args=None, expected_keys=None):
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    output = call_model_args['network'](images)
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        target = output[:,target_class_idx]
        call_model_args['network'].zero_grad()
        grads = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target))[0].squeeze()
        gradients = grads.cpu().detach().numpy()
        
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



def handles_function(images, call_model_args=None, expected_keys=None):
    
    handle_forward = call_model_args['layer'].register_forward_hook(conv_layer_forward)
    handle_backward = call_model_args['layer'].register_full_backward_hook(conv_layer_backward)

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    output = call_model_args['network'](images)
    
    target = output[:,target_class_idx]
    call_model_args['network'].zero_grad()
    _ = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
    
    handle_forward.remove()
    handle_backward.remove()
    
    return conv_layer_outputs, output



#------------------------------------------------------------------------------#

#--------------------------------     BP     ----------------------------------#



def BP(images, model):
    
    # List of saliency maps
    saliency_maps = []
    
    # For each input image we display and save the results
    for idx, img in enumerate(images):
        
        # Load input image
        im_tensor = img.unsqueeze(0).to(device)
        
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
    
        # Retrieve output from the image
        y_pred = model(im_tensor)[0]
    
        # Do backpropagation to get the derivative of the output based on the image
        model.zero_grad()
        y_pred.backward()
        grads = im_tensor.grad.data.squeeze()
        attribution_map = grads.cpu().detach().numpy()
        
        # Store map
        saliency_maps.append(attribution_map)
    
    # Return saliency maps
    return np.array(saliency_maps)



#------------------------------------------------------------------------------#

#-------------------------------     IG0     ----------------------------------#



def IG0(images, model):
    
    # List of saliency maps
    saliency_maps = []
    
    # For each input image we display and save the results
    for idx, img in enumerate(images):
        
        # Load input image and raw attributions for given slice and method
        im_tensor = img.unsqueeze(0)
        im_numpy = im_tensor.squeeze().cpu().numpy()
        
        # Launch model on input image
        y_pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
        
        # Construct the saliency object. This alone doesn't do anything.
        integrated_gradients = saliency.IntegratedGradients()
        
        # Zero-value baseline
        baseline_zero = np.zeros_like(im_numpy)
        
        call_model_args = {'network': model}
        
        # Compute Integrated Gradients for zero baseline
        attr_zero = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=25)
        
        # Invert attribution values if prediction is "without contrast agent"
        if (y_pred <= 0.5):
            attr_zero = attr_zero * -1
        
        # Store map
        saliency_maps.append(attr_zero)
    
    # Return saliency maps
    return np.array(saliency_maps)



#------------------------------------------------------------------------------#

#------------------------------     GradCAM     -------------------------------#



def GradCAM(images, model, layer):
    
    # List of saliency maps
    saliency_maps = []
    
    # Look at all layers
    for i, d in enumerate(model.named_modules()):
        
        # If selected layer
        if (d[0] == layer):
            print('\t\t Layer', d[0])
            
            # Save layer and break for loop
            conv_layer = d[1]
            break
        
    # For each input image we display and save the results
    for idx, img in enumerate(images):
        
        # Load input image and raw attributions for given slice and method
        im_tensor = img.unsqueeze(0)
        im_numpy = im_tensor.squeeze().cpu().numpy()
        
        call_model_args = {'network': model, 'layer': conv_layer}
        
        # Place handles
        x_value_batched = np.expand_dims(im_numpy, axis=0)
        data, output = handles_function(x_value_batched,
                                        call_model_args=call_model_args,
                                        expected_keys=expected_keys_GradCAM)
        
        # Exctract features
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
        if np.max(grad_cam) > 0:
            grad_cam = grad_cam / np.max(grad_cam)
        grad_cam = resize(grad_cam, im_numpy.shape[:2])
        
        # Store map
        saliency_maps.append(grad_cam)
    
    # Return saliency maps
    return np.array(saliency_maps)



#------------------------------------------------------------------------------#

#---------------------------------     EG     ---------------------------------#



def EG(images, bkgd_batches, model): 
    
    # Empty list for storing all runs results
    raw_all_runs = []
    
    # For each run (random seed)
    for rseed in range (nbRunsEG):
        print("\t\t Run n°", rseed+1)
        
        # One run attributions
        raw_all_batches = []
    
        # For each batch of test_images
        for batch in bkgd_batches:
            
            # GradientExplainer
            e = shap.GradientExplainer(model, batch)
            
            # Compute SHAP values for given examples
            shap_values = e.shap_values(images, nsamples=len(batch), rseed=rseed).squeeze()
        
            # For each test slice
            for idx, img in enumerate(images):
                
                # Launch model on input image
                im_tensor = img.unsqueeze(0)
                y_pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
                
                # Invert attribution values if label is "without contrast agent"
                if (y_pred <= 0.5): shap_values[idx] = shap_values[idx] * -1
            
            # Save as part of one run
            raw_all_batches.append(shap_values)
        
        raw_one_run = np.mean(raw_all_batches, axis=0)
        
        # Append run to list
        raw_all_runs.append(raw_one_run)
    
    # Mean of all runs
    raw_mean = np.mean(raw_all_runs, axis=0)
    
    # Return
    return raw_mean



#------------------------------------------------------------------------------#

#--------------------------     REVERSE TRANSFORM     -------------------------#



def reverse_registration(maps_transformed):
    
    # List of saliency maps
    saliency_maps_reversed = []
    
    # For each map
    for idx, attr in enumerate(maps_transformed):
        
        if (idx != reference_idx):
            
            # Apply Reverse transform
            fullpathParameters = pathTransformed + 'Parameters_' + str(idx) + '.2.txt'
            
            # Convert to ITK image
            itk_attr = itk.GetImageFromArray(attr.astype(np.float32))
            
            # Reverse parameter object
            inputParams = itk.ParameterObject.New()
            inputParams.ReadParameterFile(fullpathParameters)
        
            # Call transformix function
            attr_reversed = itk.transformix_filter(itk_attr, inputParams)
            
            # Convert to numpy array
            attr_reversed_np = np.asarray(attr_reversed)
            
            # Store reversed saliency map
            saliency_maps_reversed.append(attr_reversed_np)
            
        else:
            saliency_maps_reversed.append(attr)
     
    # Return reversedsaliency maps
    return np.array(saliency_maps_reversed)



#------------------------------------------------------------------------------#

#------------------------     SAVE REGISTERED MAPS     ------------------------#



def save_registered(maps_transformed, method):
    
    # For each map
    for idx, attr in enumerate(maps_transformed):
        if (idx > 60 and idx < 65):
            attr_abs = np.abs(attr)
            np.save(pathResults + 'Registered_ImgReg+' + method + '_Im_' +str(idx+1)+ '.npy', attr_abs)



#------------------------------------------------------------------------------#

#---------------------------------     MAIN     -------------------------------#



# Defining main function
def main(): 
    
    # Load original & transformed images
    test_images_transf = torch.load(pathImgReg + test_images_filename + '_transf.pt').to(device)
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_3rd.pt').to(device)
    bkgd_batches = torch.split(background, 40)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # Range of methods
        for method in (INTERPRET_METHODS):
            
            print('\t Method', method)
            
            maps_transformed = ()
            
            # Compute saliency maps
            if (method == 'BP'):
                maps_transformed = BP(test_images_transf, model)
            elif (method == 'IG(0)'):
                maps_transformed = IG0(test_images_transf, model)
            elif (method == 'EG'):
                maps_transformed = EG(test_images_transf, bkgd_batches, model)
            elif (method == 'GradCAM'):
                layer = networks[arch]
                maps_transformed = GradCAM(test_images_transf, model, layer)
            
            # Save registered saliency maps
            save_registered(maps_transformed, method)
            
            # Call reverse registration
            maps_reversed = reverse_registration(maps_transformed)
            
            # Save path
            pathSave = pathResults + arch + '/Raw_attrs/ImgReg+' + method + '/'
            os.makedirs(pathSave, exist_ok=True)
            pathSaveAbs = pathResults + arch + '/Raw_attrs(absolute)/ImgReg+' + method + '/'
            os.makedirs(pathSaveAbs, exist_ok=True)
            
            # Save each map separately (also in absolute)
            for idx, attr in enumerate(maps_reversed):
                np.save(pathSave + 'Raw_attr_ImgReg+' + method + '_Im_' +str(idx+1)+ '.npy', attr)
                attr_abs = np.abs(attr)
                np.save(pathSaveAbs + 'Raw_attr_ImgReg+' + method + '_Im_' +str(idx+1)+ '.npy', attr_abs)



# Using the special variable
if __name__=="__main__": 
    main()