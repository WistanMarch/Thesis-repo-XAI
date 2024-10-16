# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:58:09 2022

@author: Wistan
"""


import os
import torch
import numpy as np


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# Defining main function
def main(): 
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_290.pt').to(device)
    
    # For each input image we display and save the results
    for sliceIdx in range (len(test_images)):
        
        print("Random Slice number", sliceIdx+1)
        
        # Load input image and raw attributions for given slice and method
        im_numpy = test_images[sliceIdx].squeeze().cpu().numpy()
        
        # Random 
        random_map = np.random.uniform(-1,1,(im_numpy.shape))
        
        # For each network
        for arch in networks:
            
            # Saving path
            pathSave = pathResults + arch + '/Raw_attrs/Random/'
            
            # Raw attribution save
            os.makedirs(pathSave, exist_ok=True)
            np.save(pathSave + 'Raw_attr_Random_Im_' +str(sliceIdx+1)+ '.npy', random_map)


# Using the special variable
if __name__=="__main__": 
    main()