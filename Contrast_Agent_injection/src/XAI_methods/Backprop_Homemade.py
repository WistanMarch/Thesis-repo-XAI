# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:32:28 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import pickle


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'


# Defining main function
def main(): 
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/Raw_attrs/BP/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        
        #######     FULL BACKPROPAGATION ATTRIBUTION MAP       #######
        
        
        # For each input image we display and save the results
        for i in range (len(test_images)):
            
            print("\t Backpropagation Slice number", i+1)
            
            # Load input image
            im_tensor = test_images[i].unsqueeze(0).to(device)
            
            # Set the requires_grad_ to the image for retrieving gradients
            im_tensor.requires_grad = True
        
            # Retrieve output from the image
            y_pred = model(im_tensor)[0]
        
            # Do backpropagation to get the derivative of the output based on the image
            model.zero_grad()
            y_pred.backward()
            grads = im_tensor.grad.data.squeeze()
            attribution_map = grads.cpu().detach().numpy()
            
            # Raw attribution save
            os.makedirs(pathResults, exist_ok=True)
            np.save(pathResults + 'Raw_attr_BP_Im_' +str(i+1)+ '.npy', attribution_map)


# Using the special variable
if __name__=="__main__": 
    main()