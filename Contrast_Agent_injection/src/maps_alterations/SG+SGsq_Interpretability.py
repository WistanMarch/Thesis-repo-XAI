# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:55:04 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import pickle
import saliency.core as saliency
from captum.attr import Deconvolution
import shap
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
from skimage.transform import resize
import pandas as pd


#------------------------------------     Hooks Functions     ------------------------------------#


conv_layer_outputs = {} 
def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()
    

#------------------------------------         Utils Function        ------------------------------------#


# External function for Integrated Gradients & GradCAM
def call_model_function(images,
                        call_model_args=None,
                        expected_keys=None):

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    
    output = call_model_args['network'](images)
    target = output[:,target_class_idx]
    call_model_args['network'].zero_grad()
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        grads = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target))[0].squeeze()
        gradients = grads.cpu().detach().numpy()
        
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    
    else:
        _ = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
        
        return conv_layer_outputs, output


#------------------------------------         METHODS FUNCTIONS        ------------------------------------#


# Define Backpropagation function
def BP(samples, model, SG_type):

    # Set the requires_grad_ to the image for retrieving gradients
    samples.requires_grad = True
    target_class_idx = 0

    # Retrieve output from the image
    output = model(samples)
    target = output[:,target_class_idx]
            
    # Do backpropagation to get the derivative of the output based on the images
    model.zero_grad()
    grads = torch.autograd.grad(target, samples, grad_outputs=torch.ones_like(target))[0].squeeze()
    grads = grads.cpu().detach().numpy()
    
    # Square of gradients for SmoothGrad Squared (if required)
    if (SG_type == 'SGSQ'): grads = (grads * grads)
    
    # Divide by number of samples
    attrib = np.mean(grads, axis=0)
    
    # Return attribution map
    return attrib



# Define Deconvolution function
def Deconv(samples, model, SG_type):

    # Deconvolution init
    deconv = Deconvolution(model)
    
    # Set the requires_grad_ to the image for retrieving gradients
    samples.requires_grad = True

    # Computes Deconvolution attribution scores
    grads = deconv.attribute(inputs=samples, target=0).squeeze().detach().cpu().numpy()
    
    # Square of gradients for SmoothGrad Squared (if required)
    if (SG_type == 'SGSQ'): grads = (grads * grads)
    
    # Divide by number of samples
    attrib = np.mean(grads, axis=0)
    
    # Return attribution map
    return attrib



# Define Integrated Gradients (Black & White) function
def IG(image_np, model, SG_type, ref, noise, pred):
    
    # Construct the saliency object. This alone doesn't do anything.
    integrated_gradients = saliency.IntegratedGradients()
    
    # Baselines are both black and white images
    if (ref == 'zeros'):
        baseline = np.zeros_like(image_np)
    elif (ref == 'ones'):
        baseline = np.ones_like(image_np)
    
    call_model_args = {'network': model}
    
    # Square of gradients for SmoothGrad Squared (if required)
    if (SG_type == 'SGSQ'):
        attrib = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline, batch_size=25, magnitude=True)
    # Gradients for SmoothGrad (if required)
    else:
        attrib = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline, batch_size=25, magnitude=False)
        # Invert attribution values if prediction is "without contrast agent"
        if (pred <= 0.5):
            attrib = attrib * -1
    
    # Return attribution map
    return attrib
    
     
    
# Define Integrated Gradients (Black & White) function
def IGAdaptW0B1(image_np, model, SG_type, noise, pred):
    
    if (pred <= 0.5):
        attrib = IG(image_np, model, SG_type, 'ones', noise, pred)
    else:
        attrib = IG(image_np, model, SG_type, 'zeros', noise, pred)
    
    # Return attribution map
    return attrib

     
    
# Define Integrated Gradients (Black & White) function
def IGAdaptW1B0(image_np, model, SG_type, noise, pred):
    
    if (pred <= 0.5):
        attrib = IG(image_np, model, SG_type, 'zeros', noise, pred)
    else:
        attrib = IG(image_np, model, SG_type, 'ones', noise, pred)
    
    # Return attribution map
    return attrib

  

# Define Expected Gradients function
def EG(samples, model, SG_type, pred):

    nbRuns = 1
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_3rd.pt').to(device)
    bkgd_batches = torch.split(background, 50)
        
    # Empty list for storing all runs results
    attrib_all_runs = []
    
    # For each run (random seed)
    for rseed in range (nbRuns):
        # print("\t\t\t\t Run n°", rseed+1)
        
        # One run attributions
        raw_all_batches = []
    
        # For each batch of test_images
        for batch in bkgd_batches:
            
            # GradientExplainer
            e = shap.GradientExplainer(model, batch)
    
            # Compute SHAP values for given examples
            shap_values = e.shap_values(samples, nsamples=int(len(batch)/2), rseed=rseed).squeeze()
        
            # Invert attribution values if label is "without contrast agent"
            if (pred <= 0.5): shap_values = shap_values * -1
        
            # Save as part of one run
            raw_all_batches.append(shap_values)
        
        grads_one_run = np.mean(raw_all_batches, axis=0)
        
        # Square of gradients for SmoothGrad Squared (if required)
        if (SG_type == 'SGSQ'): grads_one_run = (grads_one_run * grads_one_run)
        
        # Divide by number of samples
        attrib_one_run = np.mean(grads_one_run, axis=0)
        
        # Append run to list
        attrib_all_runs.append(attrib_one_run)

    # Mean of all runs
    attrib = np.mean(attrib_all_runs, axis=0)
    
    # Return attribution map
    return attrib

     
    
# Define Expected Gradients function
def GradCAM(samples, model, arch, SG_type, pred):
    
    # Expected keys / parameters for GradCAM
    expected_keys = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]
    should_resize=True
    three_dims=False
    
    # Set the requires_grad_ to the image for retrieving gradients
    samples.requires_grad = True
        
    # Look at all layers
    for i, d in enumerate(model.named_modules()):
        
        # If selected layer
        if (d[0] == networks[arch]):
            
            # Model & Layer args
            conv_layer = d[1]
            call_model_args = {'network': model, 'layer': conv_layer}
            break
            
    # Final grads lists
    grads = []

    # For each sample
    for i in range (len(samples)):
        
        # Extract sample
        x_value = samples[i].squeeze().detach().cpu().numpy()
        
        # Register hooks for selected layer
        handle_forward = call_model_args['layer'].register_forward_hook(conv_layer_forward)
        handle_backward = call_model_args['layer'].register_full_backward_hook(conv_layer_backward)
        
        # GradCAM execution
        x_value_batched = np.expand_dims(x_value, axis=0) 
        data, output = call_model_function(x_value_batched,
                                           call_model_args=call_model_args,
                                           expected_keys=expected_keys)
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # Product of maps
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
            
        # Append to grads of samples
        if (SG_type == 'SGSQ'): grads.append(grad_cam * grad_cam)
        else: grads.append(grad_cam)
        
    # Divide by number of samples
    attrib = np.mean(grads, axis=0)
    
    # Return attribution map
    return attrib




#------------------         PARAMETERS        ------------------#


# Base paths
pathRoot = './'


# List of trained networks
networks = {
            "resnet" : "layer4.2.conv1",
            "Xception" : "conv4.conv1",
            }


# All methods for files loading
INTERPRET_METHODS = [
                    'BP',
                    # 'Deconv',
                    'IG(0)',
                    # 'IG(1)',
                    # 'IG(0-1)',
                    # 'IGA(01)',
                    # 'IGAW0B1',
                    'EG',
                    'GradCAM',
                     ]


# All Map Types to load
MAP_TYPES = {
            'Pix-Abs' : 'Raw_attrs(absolute)',
            'Pix-Orig' : 'Raw_attrs',
            # 'Reg-Abs' : 'XRAI_attrs(absolute)',
            # 'Reg-Orig' : 'XRAI_attrs',
             }


# SmoothGrad & SmoothGrad Squared
SG_TYPES = [
            'SG',
            'SGSQ',
            ]


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Samples Number Parameter
nsamples = 50


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#------------------         MAIN        ------------------#


# Defining main function
def main():
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # Set random seed
    np.random.seed(1)
    
    # For each network
    for arch in networks:
        print("Network " + arch)

        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # Load list of best noise levels for SG & SG²
        df_noise_lvls = pd.read_csv(pathRoot + 'Results/' + arch + '_SG_SGsq_noise_lvls.csv', index_col=0).fillna(0)
                
        # For each method
        for method in INTERPRET_METHODS:
            
            # Extract corresponding noise levels
            df_noise_method = df_noise_lvls[df_noise_lvls['Method'] == method]
        
            # If SG or SG² is used
            for SG_type in SG_TYPES:
                print("\t" + method + " " + SG_type)
                    
                # Extract corresponding noise levels
                df_noise_SG = df_noise_method[df_noise_method['SG-SGSQ'] == SG_type]
                
                # For each map type
                for map_type in MAP_TYPES:
                    print("\t\t" + map_type)
                    
                    # Extract noise level to apply
                    noise_lvl = df_noise_SG[map_type].to_list()[0]
                    
                    # Check if noise level is != than 0.0
                    if (noise_lvl != 0.0):
                        
                        # Path for saving results
                        pathResults = pathRoot + 'Results/' + arch + '/' + MAP_TYPES[map_type] + '/'
                        
                        # For each input image we display and save the results
                        for sliceIdx in range (len(test_images)):
                              
                            if (sliceIdx not in excluded_idx):
                                
                                # Display of the current slice (every 50)
                                if (sliceIdx == 0 or (sliceIdx+1) % 50 == 0 or sliceIdx == len(test_images)-1): print("\t\t\t Slice", sliceIdx+1)
                                
                                # Save name
                                filenameStart = 'Raw_attr_'
                                
                                # Load input image
                                im_tensor = test_images[sliceIdx].unsqueeze(0)
                                im_numpy = im_tensor.squeeze().cpu().numpy()
                                
                                # Launch model on input image
                                pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
                        
                                # Standard Deviation of the noise
                                stdev = noise_lvl * (np.max(im_numpy) - np.min(im_numpy))
                                
                                # Array of all noisy samples
                                all_samples = []
                                
                                # For each sample
                                for i in range(nsamples):
                                    # Apply noise to the input image
                                    noise = np.random.normal(0, stdev, im_numpy.shape)
                                    x_plus_noise = im_numpy + noise
                                    # Add sample to array
                                    all_samples.append(x_plus_noise)
                    
                                # Convert to tensor for method
                                all_samples_tensor = torch.tensor(all_samples, device=device, dtype=torch.float32).unsqueeze(1)
            
                        ### METHODS CALLS
            
                                # Launch method function (different input for IG variants)
                                if (method == 'BP'):
                                    grads = BP(all_samples_tensor, model, SG_type)
                                    
                                elif (method == 'Deconv'):
                                    grads = Deconv(all_samples_tensor, model, SG_type)
                                    
                                elif (method == 'GradCAM'):
                                    grads = GradCAM(all_samples_tensor, model, arch, SG_type, pred)

                                elif (method == 'EG'):
                                    grads = EG(all_samples_tensor, model, SG_type, pred)
                                    
                                elif (method == 'IG(0)'):
                                    grads = IG(im_numpy, model, SG_type, 'zeros', noise_lvl, pred)
                                    
                                # elif (method == 'IG(1)'):
                                #     grads = IG(im_numpy, model, SG_type, 'ones', noise_lvl, pred)
                                    
                                # elif (method == 'IG(0-1)'):
                                #     grads_B = IG(im_numpy, model, SG_type, 'zeros', noise_lvl, pred)
                                #     grads_W = IG(im_numpy, model, SG_type, 'ones', noise_lvl, pred)
                                #     grads = np.mean([grads_B, grads_W], axis=0)
                                    
                                # elif (method == 'IGAW0B1'):
                                #     grads = IGAdaptW0B1(im_numpy, model, SG_type, noise_lvl, pred)
                                    
                                # elif (method == 'IGAW1B0'):
                                #     grads = IGAdaptW1B0(im_numpy, model, SG_type, noise_lvl, pred)
                                
                        ### END METHODS CALLS
                                    
                                # Convert to absolute if needed
                                if ('Abs' in map_type):
                                    grads = np.abs(grads)
                                
                                # Segment if needed
                                if ('Reg' in map_type):
                                    # Change file name start
                                    filenameStart = 'XRAI_attr_'
                                    # Construct the saliency object. This alone doesn't do anything.
                                    xrai_object = saliency.XRAI()
                                    # Launch XRAI with attribution map
                                    grads = xrai_object.GetMask(x_value=im_numpy, call_model_function=None, base_attribution=grads)
                                    
                                # Full file path
                                methodNewName = SG_type + '+' + method
                                pathSave = pathResults + methodNewName + '/'
                                os.makedirs(pathSave, exist_ok=True)
                                
                                # Save attribution map
                                np.save(pathSave + filenameStart + methodNewName + '_Im_' + str(sliceIdx+1) + '.npy', grads)

    

# Using special variable
if __name__=="__main__":
    main()