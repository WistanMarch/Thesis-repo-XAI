# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import time
import torch
import numpy as np
import pickle
import saliency.core as saliency


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Raw_attrs/'
    
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = pickle.load(open(pathRoot + 'serialized_param_predictor.sav', 'rb')).to(device)


def call_model_function(images, call_model_args=None, expected_keys=None):
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_branch = call_model_args[0]
    target_idx = call_model_args[1]
    output = model(images)
    pred_branch = torch.sigmoid(output[target_branch])
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = pred_branch[:,target_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = grads[0].squeeze(axis=1)
        gradients = grads.cpu().detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}



# Defining main function
def main(): 

    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_263.pt').to(device)
    # Load params dictionnary
    f = open("./params.pkl", "rb")
    params = pickle.load(f)
    f.close()
    
    
    #######     FULL INTEGRATED GRADIENTS ATTRIBUTION MAP       #######
    
    
    # For each input image we display and save the results
    for sliceIdx in range (len(test_images)):
        
        print("Integrated Gradients Slice number", sliceIdx+1)
        
        # Load input image and raw attributions for given slice and method
        im_tensor = test_images[sliceIdx].unsqueeze(0)
        im_numpy = im_tensor.squeeze().cpu().numpy()
        
        # Retrieve output from the image
        y_pred = model(im_tensor.to(device))
    
        # For each parameter prediction
        for param_name, pred in y_pred.items():
        
            print("\t Param", param_name)
            
            # Apply sigmoid on branch prediction
            pred = torch.sigmoid(pred)
            # Find maximum, index, and class of prediction
            idx_max_pred = pred.argmax()
            max_pred = pred[0, idx_max_pred]
            pred_class = params[param_name][idx_max_pred.detach().cpu().numpy()]
        
            # Construct the saliency object. This alone doesn't do anything.
            integrated_gradients = saliency.IntegratedGradients()
            
            # Baselines are both black and white images
            baseline_black = np.zeros_like(im_numpy)
            baseline_white = np.ones_like(im_numpy)
            
            # Compute Integrated Gradients for each baseline
            attr_black = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, call_model_args=[param_name, idx_max_pred], x_baseline=baseline_black, x_steps=200)
            attr_white = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, call_model_args=[param_name, idx_max_pred], x_baseline=baseline_white, x_steps=200)
            # Create a combined attribution map (average of black and white)
            attrs = [attr_black, attr_white]
            attrs_mean = np.mean(attrs, axis=0)
            
            # Raw attribution save
            if (param_name == 'contrast_used'):
                np.save(pathResults + param_name + '/IGB/IGB_Im_' +str(sliceIdx+1)+ '_pred_' + str(max_pred.detach().cpu().numpy()) + '.npy', attr_black)
                np.save(pathResults + param_name + '/IGW/IGW_Im_' +str(sliceIdx+1)+ '_pred_' + str(max_pred.detach().cpu().numpy()) + '.npy', attr_white)
                np.save(pathResults + param_name + '/IGBW/IGBW_Im_' +str(sliceIdx+1)+ '_pred_' + str(max_pred.detach().cpu().numpy()) + '.npy', attrs_mean)
            else:
                np.save(pathResults + param_name + '/IGB/IGB_Im_' +str(sliceIdx+1)+ '_pred_' + str(pred_class) + '.npy', attr_black)
                np.save(pathResults + param_name + '/IGW/IGW_Im_' +str(sliceIdx+1)+ '_pred_' + str(pred_class) + '.npy', attr_white)
                np.save(pathResults + param_name + '/IGBW/IGBW_Im_' +str(sliceIdx+1)+ '_pred_' + str(pred_class) + '.npy', attrs_mean)
    
            # Sleep to allow GPU cooling
            time.sleep(2)


# Using the special variable
if __name__=="__main__": 
    main()