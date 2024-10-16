# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:04:18 2024

@author: Wistan
"""


"""
Some maps already existed, and were simply transferred to the results folder :
    - Deconv custom
    - Deconv Captum
    - IGO PAIR
    - GradCAM PAIR
"""



import os
import torch
import numpy as np
import pickle
from skimage.transform import resize

import shap
import saliency.core as saliency
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam



#--------------------------------     PARAMS     --------------------------------#



# List of trained networks
networks = {
            "resnet" : "layer4.2.conv1",
            }


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#--------------------------------------------------------------------------------#

#--------------------------------     UTILS     ---------------------------------#



def call_model_function(images, call_model_args=None, expected_keys=None):
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    output = call_model_args['network'](images)
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        target = output[0,target_class_idx]
        call_model_args['network'].zero_grad()
        target.backward()
        grads = images.grad.data.squeeze(axis=1)
        gradients = grads.cpu().detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



#--------------------------------------------------------------------------------#

#----------------------------------     BP     ----------------------------------#



# Custom BP
def BP_custom(pathOutNet, images, model):
    
    pathOut = pathOutNet + '/BP_custom/'
    
    print("\t BP custom")
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # Load input image
        im_tensor = images[i].unsqueeze(0).to(device)
        
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
    
        # Retrieve output from the image
        y_pred = model(im_tensor)[0]
    
        # Do backpropagation to get the derivative of the output based on the image
        model.zero_grad()
        y_pred.backward()
        grads = im_tensor.grad.data.squeeze()
        attr = grads.cpu().detach().numpy()
        
        # To absolute values
        attr_abs = np.abs(attr)
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_BP_custom_Im_' +str(i+1)+ '.npy', attr_abs)



# Captum BP
def BP_captum(pathOutNet, images, model):
    
    pathOut = pathOutNet + '/BP_captum/'
    
    print("\t BP captum")
    
    # Captum BP init
    saliency = Saliency(model)
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # Load input image
        im_tensor = images[i].unsqueeze(0).to(device)
        
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
        
        # Computes saliency map
        attr = saliency.attribute(inputs=im_tensor, target=0, abs=True)
        
        # To numpy array
        attr_np = attr.squeeze().detach().cpu().numpy()
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_BP_captum_Im_' +str(i+1)+ '.npy', attr_np)



# PAIR BP
def BP_pair(pathOutNet, images, model):
    
    pathOut = pathOutNet + '/BP_pair/'
    
    print("\t BP PAIR")
    
    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # Convert image to numpy
        im_np = images[i].squeeze().cpu().numpy()
        
        call_model_args = {'network': model}
        
        # Compute the vanilla mask
        attr = gradient_saliency.GetMask(im_np, call_model_function, call_model_args)
        
        # To absolute values
        attr_abs = np.abs(attr)
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_BP_pair_Im_' +str(i+1)+ '.npy', attr_abs)



#--------------------------------------------------------------------------------#

#----------------------------------     IG0     ---------------------------------#



# Captum IG0
def IG0_captum(pathOutNet, images, model):
    
    pathOut = pathOutNet + '/IG0_captum/'
    
    # Integrated Gradients init
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        print("\t IG0 Captum", i+1)
        
        # Load input image
        im_tensor = images[i].unsqueeze(0).to(device)
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
        
        # Black baseline
        baseline_black = torch.zeros(size=im_tensor.shape, device=device, requires_grad=True)
        
        # Computes integrated gradients
        attr = ig.attribute(inputs=im_tensor, baselines=baseline_black, target=0, n_steps=200, method='gausslegendre', internal_batch_size=25, return_convergence_delta=False)
        
        # To absolute & numpy array
        attr_np = np.abs(attr.squeeze().detach().cpu().numpy())
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_IG0_captum_Im_' +str(i+1)+ '.npy', attr_np)



#--------------------------------------------------------------------------------#

#----------------------------------     EG     ----------------------------------#



# shap EG
def EG_shap(pathOutNet, images, bkgd_batches, model):
    
    pathOut = pathOutNet + '/EG_shap/'
    
    print("\t EG shap")
    
    # One run attributions
    raw_all_batches = []

    # For each batch of test_images
    for batch in bkgd_batches:
        
        # GradientExplainer
        print("\t\t Start Explainer...")
        e = shap.GradientExplainer(model, batch)
        print("\t\t End Explainer...")
        
        # Compute SHAP values for given examples
        print("\t\t Start Application...")
        shap_values = e.shap_values(images, nsamples=len(batch), rseed=42).squeeze()
        print("\t\t End Application...")
        
        # To absolute values
        shap_values_abs = np.abs(shap_values)
        
        # Save as part of one run
        raw_all_batches.append(shap_values_abs)
    
    # Average across custom batches
    all_attrs = np.mean(raw_all_batches, axis=0)
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # Extract specific image attr map
        attr = all_attrs[i]
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_EG_shap_Im_' +str(i+1)+ '.npy', attr)



# Captum EG
def EG_captum(pathOutNet, images, bkgd_batches, model):
    
    pathOut = pathOutNet + '/EG_captum/'
    
    print("\t EG captum")
    
    # Expected Gradients init
    gradient_shap = GradientShap(model, multiply_by_inputs=True)
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # One run attributions
        raw_all_batches = []
    
        # For each batch of test_images
        for batch in bkgd_batches:
            
            # Computes gradient shap for the input
            shap_values = gradient_shap.attribute(inputs=images[i].unsqueeze(0), baselines=batch, n_samples=len(batch), target=0, return_convergence_delta=False)
            
            # To absolute values
            shap_values_abs = np.abs(shap_values.squeeze().detach().cpu().numpy())
            
            # Save as part of one run
            raw_all_batches.append(shap_values_abs)
        
        # Average across custom batches
        attr_abs = np.mean(raw_all_batches, axis=0)
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_EG_captum_Im_' +str(i+1)+ '.npy', attr_abs)



#--------------------------------------------------------------------------------#

#-------------------------------     GRADCAM     --------------------------------#



# Captum GradCAM
def GradCAM_captum(pathOutNet, images, model, layer):
    
    pathOut = pathOutNet + '/GradCAM_captum/'
    
    print("\t GradCAM Captum")
    
    # GradCAM object init
    gradcam = LayerGradCam(model, layer)
    
    # For each input image we display and save the results
    for i in range (len(images)):
        
        # Load input image
        im_tensor = images[i].unsqueeze(0).to(device)
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
        
        # Numpy input image
        im_np = im_tensor.squeeze().detach().cpu().numpy()
        
        # Launch model on input image
        y_pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
        
        # Compute attr
        attr = gradcam.attribute(inputs=im_tensor, target=0, attribute_to_layer_input=False, relu_attributions=False)
        
        # To numpy array
        attr_np = attr.squeeze().detach().cpu().numpy()
        
        # Remove either positive or negative attrs depending on predicted class
        if (y_pred <= 0.5):
            attr_np = np.abs(np.minimum(attr_np, 0))
        else:
            attr_np = np.maximum(attr_np, 0)
        
        # Resize
        if np.max(attr_np) > 0:
            attr_np = attr_np / np.max(attr_np)
        attr_np = resize(attr_np, im_np.shape[:2])
        
        # Raw attribution save
        os.makedirs(pathOut, exist_ok=True)
        np.save(pathOut + 'Raw_attr_GradCAM_captum_Im_' +str(i+1)+ '.npy', attr_np)



#--------------------------------------------------------------------------------#

#---------------------------------     MAIN     ---------------------------------#



def main():

    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_30.pt').to(device)
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_2nd.pt').to(device)
    bkgd_batches = torch.split(background, 50)
    
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # Output path for network
        pathOutNet = pathResults + arch
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        model.eval()
        
        # Look at all layers
        for i, d in enumerate(model.named_modules()):
            # If selected layer
            if (d[0] == networks[arch]):
                # Save layer and break for loop
                layer = d[1]
                break
        
        
        ### CALL METHODS
        
        BP_custom(pathOutNet, test_images, model)
        BP_captum(pathOutNet, test_images, model)
        BP_pair(pathOutNet, test_images, model)
        
        IG0_captum(pathOutNet, test_images, model)
        
        EG_shap(pathOutNet, test_images, bkgd_batches, model)
        EG_captum(pathOutNet, test_images, bkgd_batches, model)
        
        GradCAM_captum(pathOutNet, test_images, model, layer)
    



# Using the special variable
if __name__=="__main__": 
    main()