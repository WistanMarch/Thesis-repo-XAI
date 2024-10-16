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



# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


pathRoot = './'
pathResults = pathRoot + 'Results/'


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def standardize_image(im):
    extreme_value = np.max(np.abs(im))
    return im / extreme_value



def call_model_function(images, call_model_args=None, expected_keys=None):
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    target_class_idx = 0
    output = call_model_args['network'](images)
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        target = output[:,target_class_idx]
        grads = torch.autograd.grad(target, images, grad_outputs=torch.ones_like(target))[0].squeeze()
        gradients = grads.cpu().detach().numpy()
        
        print(np.mean(gradients, axis=(1,2)))
        
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        


# Defining main function
def main(): 
    
    # Load images tensor (to keep the same bkgd and test images)
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResultsIGB = pathResults + arch + '/Raw_attrs/IG(0)/'
        pathResultsIGW = pathResults + arch + '/Raw_attrs/IG(1)/'
        pathResultsIGBW = pathResults + arch + '/Raw_attrs/IG(0-1)/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        model.eval()
      
    
        #######     FULL INTEGRATED GRADIENTS ATTRIBUTION MAP       #######
        
        
        # For each input image we display and save the results
        for sliceIdx in range (len(test_images)):
            
            print("\t Integrated Gradients Image n°", sliceIdx+1)
            
            # Load input image and raw attributions for given slice and method
            im_tensor = test_images[sliceIdx].unsqueeze(0)
            im_numpy = im_tensor.squeeze().cpu().numpy()
            
            # Launch model on input image
            y_pred = torch.sigmoid(model(im_tensor)).detach().cpu().numpy()[0][0]
            
            # Construct the saliency object. This alone doesn't do anything.
            integrated_gradients = saliency.IntegratedGradients()
            
            # Baselines are both black and white images
            baseline_black = np.zeros_like(im_numpy)
            baseline_white = np.ones_like(im_numpy)
            
            call_model_args = {'network': model}
            
            # Compute Integrated Gradients for each baseline
            attr_black = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_black, x_steps=200, batch_size=25)
            attr_white = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_white, x_steps=200, batch_size=25)
            # Standardize both maps & create an averaged attribution map
            attr_black_stand = standardize_image(attr_black)
            attr_white_stand = standardize_image(attr_white)
            attrs_mean = np.mean([attr_black_stand, attr_white_stand], axis=0)
            
            # Invert attribution values if prediction is "without contrast agent"
            if (y_pred <= 0.5):
                attr_black = attr_black * -1
                attr_white = attr_white * -1
                attrs_mean = attrs_mean * -1
            
            # Raw attributions save
            os.makedirs(pathResultsIGB, exist_ok=True)
            os.makedirs(pathResultsIGW, exist_ok=True)
            os.makedirs(pathResultsIGBW, exist_ok=True)
            np.save(pathResultsIGB + 'Raw_attr_IG(0)_Im_' +str(sliceIdx+1)+ '.npy', attr_black)
            np.save(pathResultsIGW + 'Raw_attr_IG(1)_Im_' +str(sliceIdx+1)+ '.npy', attr_white)
            np.save(pathResultsIGBW + 'Raw_attr_IG(0-1)_Im_' +str(sliceIdx+1)+ '.npy', attrs_mean)
    
    

# Using the special variable
if __name__=="__main__": 
    main()