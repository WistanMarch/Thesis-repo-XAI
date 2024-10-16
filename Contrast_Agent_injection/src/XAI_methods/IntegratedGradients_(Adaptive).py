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
import time


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


pathRoot = './'


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Defining main function
def main(): 
    
    # Load images tensor (to keep the same bkgd and test images)
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResultsW0B1 = pathRoot + 'Results/' + arch + '/Raw_attrs/IGA(10)/'
        pathResultsW1B0 = pathRoot + 'Results/' + arch + '/Raw_attrs/IGA(01)/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        model.eval()
      
    
        #######     FULL INTEGRATED GRADIENTS ATTRIBUTION MAP       #######
        
        
        # For each input image we display and save the results
        for sliceIdx in range (len(test_images)):
            
            print("\t Integrated Gradients Adaptive Slice number", sliceIdx+1)
            
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
    
            # Invert attribution values if prediction is "without contrast agent"
            if (y_pred <= 0.5):
                attr_black = attr_black * -1
                attr_white = attr_white * -1
            
                # Raw attributions save
                os.makedirs(pathResultsW0B1, exist_ok=True)
                os.makedirs(pathResultsW1B0, exist_ok=True)
                np.save(pathResultsW1B0 + 'Raw_attr_IGA(01)_Im_' +str(sliceIdx+1)+ '.npy', attr_black)
                np.save(pathResultsW0B1 + 'Raw_attr_IGA(10)_Im_' +str(sliceIdx+1)+ '.npy', attr_white)
    
            # If label is "with contrast agent"
            else:
                # Raw attributions save
                os.makedirs(pathResultsW0B1, exist_ok=True)
                os.makedirs(pathResultsW1B0, exist_ok=True)
                np.save(pathResultsW0B1 + 'Raw_attr_IGA(10)_Im_' +str(sliceIdx+1)+ '.npy', attr_black)
                np.save(pathResultsW1B0 + 'Raw_attr_IGA(01)_Im_' +str(sliceIdx+1)+ '.npy', attr_white)
    
            # Sleep to allow GPU cooling
            time.sleep(2)
    

# Using the special variable
if __name__=="__main__": 
    main()