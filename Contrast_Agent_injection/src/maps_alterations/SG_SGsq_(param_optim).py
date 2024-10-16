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
        # Compute gradients using .backward
        _ = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target), retain_graph=True)[0].squeeze()
        
        return conv_layer_outputs, output


#------------------------------------         METHODS FUNCTIONS        ------------------------------------#


# Define Backpropagation function
def BP(samples, model, pathResults, sliceIdx, noise, pred):

    print("\t\t\t BP")
    
    # Set the requires_grad_ to the image for retrieving gradients
    samples.requires_grad = True
    target_class_idx = 0

    # Retrieve output from the image
    output = model(samples)
    
    # Do backpropagation to get the derivative of the output based on the images
    target = output[:,target_class_idx]
    model.zero_grad()
    grads = torch.autograd.grad(target, samples, grad_outputs=torch.ones_like(target))[0].squeeze()
    grads = grads.cpu().detach().numpy()
    
    # Square of gradients for SmoothGrad Squared
    grads_SQ = (grads * grads)
    
    # Divide by number of samples
    attrib_SG = np.mean(grads, axis=0)
    attrib_SGSQ = np.mean(grads_SQ, axis=0)
    
    # Save attributions
    os.makedirs(pathResults + 'BP/', exist_ok=True)
    np.save(pathResults + 'BP/Raw_attr_SG+BP_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SG)
    np.save(pathResults + 'BP/Raw_attr_SGSQ+BP_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SGSQ)



# Define Deconvolution function
def Deconv(samples, model, pathResults, sliceIdx, noise, pred):

    print("\t\t\t Deconv")
    
    # Deconvolution init
    deconv = Deconvolution(model)
    
    # Set the requires_grad_ to the image for retrieving gradients
    samples.requires_grad = True

    # Computes Deconvolution attribution scores
    grads = deconv.attribute(inputs=samples, target=0).squeeze().detach().cpu().numpy()
    
    # Square of gradients for SmoothGrad Squared
    grads_SQ = (grads * grads)

    # Divide by number of samples
    attrib_SG = np.mean(grads, axis=0)
    attrib_SGSQ = np.mean(grads_SQ, axis=0)
    
    # Raw attribution save
    os.makedirs(pathResults + 'Deconv/', exist_ok=True)
    np.save(pathResults + 'Deconv/Raw_attr_SG+Deconv_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SG)
    np.save(pathResults + 'Deconv/Raw_attr_SGSQ+Deconv_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SGSQ)



# Define Integrated Gradients (Black & White) function
def IG(image_np, model, pathResults, sliceIdx, noise, pred):
    
    print("\t\t\t IG (0)")
    # print("\t\t\t IG (0/1/0-1)")
    
    # Construct the saliency object. This alone doesn't do anything.
    integrated_gradients = saliency.IntegratedGradients()
    
    # Baselines are both black and white images
    baseline_zero = np.zeros_like(image_np)
    # baseline_one = np.ones_like(image_np)
    
    call_model_args = {'network': model}
    
    # Compute SmoothGrad Integrated Gradients for each baseline
    attrib_0_SG = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline_zero, batch_size=25, magnitude=False)
    # attrib_1_SG = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline_one, batch_size=25, magnitude=False)
    # # Create a combined attribution map (average of black and white)
    # attrib_mean_SG = np.mean([attrib_0_SG, attrib_1_SG], axis=0)
    
    # Invert attribution values if prediction is "without contrast agent"
    if (pred <= 0.5):
        attrib_0_SG = attrib_0_SG * -1
        # attrib_1_SG = attrib_1_SG * -1
        # attrib_mean_SG = attrib_mean_SG * -1
    
    # Compute SmoothGrad Squared Integrated Gradients for each baseline
    attrib_0_SGSQ = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline_zero, batch_size=25, magnitude=True)
    # attrib_1_SGSQ = integrated_gradients.GetSmoothedMask(x_value=image_np, call_model_function=call_model_function, call_model_args=call_model_args, stdev_spread=noise, nsamples=nsamples, x_steps=50, x_baseline=baseline_one, batch_size=25, magnitude=True)
    # # Create a combined attribution map (average of black and white)
    # attrib_mean_SGSQ = np.mean([attrib_0_SGSQ, attrib_1_SGSQ], axis=0)
    
    # Raw attribution save
    os.makedirs(pathResults + 'IG(0)/', exist_ok=True)
    # os.makedirs(pathResults + 'IG(1)/', exist_ok=True)
    # os.makedirs(pathResults + 'IG(0-1)/', exist_ok=True)
    np.save(pathResults + 'IG(0)/Raw_attr_SG+IG(0)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_0_SG)
    np.save(pathResults + 'IG(0)/Raw_attr_SGSQ+IG(0)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_0_SGSQ)
    # np.save(pathResults + 'IG(1)/Raw_attr_SG+IG(1)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_1_SG)
    # np.save(pathResults + 'IG(1)/Raw_attr_SGSQ+IG(1)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_1_SGSQ)
    # np.save(pathResults + 'IG(0-1)/Raw_attr_SG+IG(0-1)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_mean_SG)
    # np.save(pathResults + 'IG(0-1)/Raw_attr_SGSQ+IG(0-1)_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_mean_SGSQ)
    
    # # If we need the Adaptive version of Integrated Gradients
    # if (IGAdaptive):
    #     print("\t\t\t Integrated Gradients (Adaptive)")
        
    #     os.makedirs(pathResults + 'IGAW1B0/', exist_ok=True)
    #     os.makedirs(pathResults + 'IGAW0B1/', exist_ok=True)
    
    #     # If label is "without contrast agent"
    #     if (pred <= 0.5):
    #         # Raw attribution save
    #         np.save(pathResults + 'IGAW1B0/Raw_attr_SG+IGAW1B0_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_bk_SG)
    #         np.save(pathResults + 'IGAW1B0/Raw_attr_SGSQ+IGAW1B0_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_bk_SGSQ)
    #         np.save(pathResults + 'IGAW0B1/Raw_attr_SG+IGAW0B1_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_wh_SG)
    #         np.save(pathResults + 'IGAW0B1/Raw_attr_SGSQ+IGAW0B1_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_wh_SGSQ)
    
    #     # If label is "with contrast agent"
    #     else:
    #         # Raw attribution save
    #         np.save(pathResults + 'IGAW0B1/Raw_attr_SG+IGAW0B1_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_bk_SG)
    #         np.save(pathResults + 'IGAW0B1/Raw_attr_SGSQ+IGAW0B1_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_bk_SGSQ)
    #         np.save(pathResults + 'IGAW1B0/Raw_attr_SG+IGAW1B0_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_wh_SG)
    #         np.save(pathResults + 'IGAW1B0/Raw_attr_SGSQ+IGAW1B0_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_wh_SGSQ)

     
    
# Define Expected Gradients function
def EG(samples, model, pathResults, sliceIdx, noise, pred):

    print("\t\t\t EG")
    
    nbRuns = 1
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_3rd.pt').to(device)
    bkgd_batches = torch.split(background, 50)
        
    # Empty list for storing all runs results
    attrib_SG_all_runs = []
    attrib_SGSQ_all_runs = []
    
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
        
        # Square of gradients for SmoothGrad Squared
        grads_SQ_one_run = (grads_one_run * grads_one_run)
        
        # Divide by number of samples
        attrib_SG_one_run = np.mean(grads_one_run, axis=0)
        attrib_SGSQ_one_run = np.mean(grads_SQ_one_run, axis=0)
        
        # Append run to list
        attrib_SG_all_runs.append(attrib_SG_one_run)
        attrib_SGSQ_all_runs.append(attrib_SGSQ_one_run)
    
    # Mean of all runs
    attrib_SG = np.mean(attrib_SG_all_runs, axis=0)
    attrib_SGSQ = np.mean(attrib_SGSQ_all_runs, axis=0)
    
    # Save final SG & SGSQ attributions
    os.makedirs(pathResults + 'EG/', exist_ok=True)
    np.save(pathResults + 'EG/Raw_attr_SG+EG_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SG)
    np.save(pathResults + 'EG/Raw_attr_SGSQ+EG_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SGSQ)

     
    
# Define Expected Gradients function
def GradCAM(samples, model, pathResults, sliceIdx, noise, pred):

    print("\t\t\t GradCAM")

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
            
            # Final grads lists
            grads = []
            grads_SQ = []

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
                grads.append(grad_cam)
                grads_SQ.append(grad_cam * grad_cam)
                
            # Divide by number of samples
            attrib_SG = np.mean(grads, axis=0)
            attrib_SGSQ = np.mean(grads_SQ, axis=0)
        
            # Raw attributions save
            os.makedirs(pathResults + 'GradCAM/', exist_ok=True)
            np.save(pathResults + 'GradCAM/Raw_attr_SG+GradCAM_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SG)
            np.save(pathResults + 'GradCAM/Raw_attr_SGSQ+GradCAM_Im_' +str(sliceIdx+1)+ '_' +str(noise)+ '.npy', attrib_SGSQ)



#------------------         PARAMETERS        ------------------#


# Base paths
pathRoot = './'


# List of trained networks
networks = {
            "resnet" : "layer4.2.conv1",
            "Xception" : "conv4.conv1",
            }


# All methods for files loading
INTERPRET_METHODS = {
                    'Backpropagation' : BP,
                    # 'Deconvolution' : Deconv,
                    'GradCAM' : GradCAM,
                    'IntegratedGradients' : IG,
                    'ExpectedGradients' : EG,
                     }


# Add Integrated Gradients (Adaptive) or not
IGAdaptive = False


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Standard Deviation Range
stdev_spread_range = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
# Samples Number Parameter
nsamples = 10


#------------------         MAIN        ------------------#


if __name__=="__main__": 

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # Set random seed
    np.random.seed(1)
    
    
    #######     STANDARD DEVIATION OPTIMIZATION       #######
    
    
    # For each input image we display and save the results
    for sliceIdx in range (len(test_images)):
                        
        if (sliceIdx not in excluded_idx):
            print("SG & SG² Image", sliceIdx+1)
            
            # Load input image
            im_tensor = test_images[sliceIdx].unsqueeze(0)
            im_numpy = im_tensor.squeeze().cpu().numpy()
            
            # For each Standard Deviation value of the range
            for noise_lvl in stdev_spread_range:
            
                print("\t STD at", noise_lvl*100, "%")
                
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
                
                # Convert to tensor for BP
                all_samples_tensor = torch.tensor(all_samples, device=device, dtype=torch.float32).unsqueeze(1)
                
                # For each network
                for arch in networks:
                    print("\t\t Network " + arch)
        
                    # Path for saving results
                    pathResults = pathRoot + 'Results/' + arch + '/SG_SGSQ_Optim/Raw_attrs/'
                    
                    # Load model
                    model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
            
                    # Launch model on input image
                    pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
                        
                    # For each method
                    for method in INTERPRET_METHODS:
                        
                        # Launch method function (different input for IG variants)
                        if (method == 'IntegratedGradients'):
                            grads = INTERPRET_METHODS[method](im_numpy, model, pathResults, sliceIdx, noise_lvl, pred)
                        else:
                            grads = INTERPRET_METHODS[method](all_samples_tensor, model, pathResults, sliceIdx, noise_lvl, pred)
                        