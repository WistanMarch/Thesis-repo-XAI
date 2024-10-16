# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:05:00 2021

@author: Wistan
"""


import os
import torch
from captum.attr import Deconvolution

import numpy as np
import pickle


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


pathRoot = './'



# Defining main function
def main(): 
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images tensor (to keep the same bkgd and test images)
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/Raw_attrs/Deconv/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        model.eval()
    
        # Deconvolution init
        deconv = Deconvolution(model)
        
        
        # For each input image we display and save the results
        for sliceIdx in range (len(test_images)):
            
            print("\t Slice number", sliceIdx+1)
            
            # Load input image
            im_tensor = test_images[sliceIdx].unsqueeze(0).to(device)
            # Set the requires_grad_ to the image for retrieving gradients
            im_tensor.requires_grad = True
        
            # Computes Deconvolution attribution scores
            attr_Deconv = deconv.attribute(inputs=im_tensor, target=0)
            
            # To numpy array
            attr_Deconv_numpy = attr_Deconv.squeeze().detach().cpu().numpy()
            
            # Raw attribution save
            os.makedirs(pathResults, exist_ok=True)
            np.save(pathResults + 'Raw_attr_Deconv_Im_' +str(sliceIdx+1)+ '.npy', attr_Deconv_numpy)
        


# Using the special variable
if __name__=="__main__": 
    main()
