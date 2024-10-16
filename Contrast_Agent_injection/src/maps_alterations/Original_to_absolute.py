# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:45:07 2022

@author: Wistan
"""


import os
import numpy as np
import torch


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'


# All methods for files loading
INTERPRET_METHODS = [
                        # 'BP',
                        # 'Deconv',
                        'IG(0)',
                        'IG(1)',
                        'IG(0-1)',
                        'IGA(01)',
                        'IGA(10)',
                       #  'EG',
                       # 'GradCAM',
                      # 'Random',
                     ]


# Defining main function
def main(): 

    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt')
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/'
        pathOriginalAttr = pathResults + 'Raw_attrs/'
        pathAbsoluteAttr = pathResults + 'Raw_attrs(absolute)/'
    
        # Range of number of test slices
        for sliceIdx in range (len(test_images)):
            print('\t Conversion to Absolute Slice number', sliceIdx+1)
        
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                # Create full attributions path
                pathLoad = pathOriginalAttr + method + '/'
                
                # Load original values results
                original_attr = np.load(pathLoad + 'Raw_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy')
                
                # Convert to absolute values
                absolute_attr = np.abs(original_attr)
                
                # Save absolute values results
                os.makedirs(pathAbsoluteAttr + method + '/', exist_ok=True)
                np.save(pathAbsoluteAttr + method + '/Raw_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy', absolute_attr)            
                
# Using the special variable
if __name__=="__main__": 
    main()