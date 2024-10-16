# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:19:12 2024

@author: Wistan
"""


import os
import torch
import numpy as np


#-----------------------------     Parameters     -----------------------------#



# Base paths
pathRoot = './'


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# All methods for files loading
INTERPRET_METHODS = [
                        # 'BP',
                        'IG(0)',
                        'GradCAM',
                        # 'EG',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
             }



#------------------------------------------------------------------------------#

#-------------------------------     UTILS     --------------------------------#



# Normalization for positive-only maps
def normalize_absolute(image):
    
    image_norm = image / np.max(image)
    return image_norm



#------------------------------------------------------------------------------#

#--------------------         AVERAGED COMBINATION         --------------------#



def average_methods(test_images, pathFolder):
    
    # For each input image
    for idx_im, _ in enumerate(test_images):
        
        # Create empty list for saliency maps
        attrs = []
        # Declare beginning of final method name
        outputName = 'Avg('
        
        # For each method
        for method in INTERPRET_METHODS:
            
            # Load map
            attr = np.load(pathFolder + method + '/' + 'Raw_attr_' + method + '_Im_' + str(idx_im+1) + '.npy')
            # Normalize
            attr_norm = normalize_absolute(attr)
            # Add to list
            attrs.append(attr_norm)
            # Add method to final name
            outputName += method + '.'
        
        # Average maps in list
        attrs_array = np.array(attrs)
        attrs_avg = np.mean(attrs_array, axis=0)
        
        # Declare final path
        outputName = outputName[:-1] + ')'
        pathOutput = pathFolder + outputName + '/'
        os.makedirs(pathOutput, exist_ok=True)
        # Save average map with final path and final name
        np.save(pathOutput + 'Raw_attr_' + outputName + '_Im_' + str(idx_im+1) + '.npy', attrs_avg)



#------------------------------------------------------------------------------#

#-------------------         WEIGHTED COMBINATION         ---------------------#



def product_methods(test_images, pathFolder):
    
    # For each input image
    for idx_im, _ in enumerate(test_images):
        
        # Create empty list for saliency maps
        attrs = []
        # Declare beginning of final method name
        outputName = 'Product('
        
        # For each method
        for method in INTERPRET_METHODS:
            
            # Load map
            attr = np.load(pathFolder + method + '/' + 'Raw_attr_' + method + '_Im_' + str(idx_im+1) + '.npy')
            # Normalize
            attr_norm = normalize_absolute(attr)
            # Add to list
            attrs.append(attr_norm)
            # Add method to final name
            outputName += method + '.'
        
        # Average maps in list
        attrs_array = np.array(attrs)
        attrs_weight = np.prod(attrs_array, axis=0)
        
        # Declare final path
        outputName = outputName[:-1] + ')'
        pathOutput = pathFolder + outputName + '/'
        os.makedirs(pathOutput, exist_ok=True)
        # Save average map with final path and final name
        np.save(pathOutput + 'Raw_attr_' + outputName + '_Im_' + str(idx_im+1) + '.npy', attrs_weight)



#------------------------------------------------------------------------------#

#--------------------------------     MAIN     --------------------------------#



# Defining main function
def main():
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').numpy()
    
    # For each network
    for arch in networks:
        
        print("Network " + arch)
        
        # For each map type
        for map_type in MAP_TYPES:
            
            print("\t" + map_type)
            
            # Input / output path
            pathFolder = pathRoot + 'Results/' + arch + '/' + MAP_TYPES[map_type] + '/'
            
            # # Apply average function
            # average_methods(test_images, pathFolder)
            
            # Apply weight function
            product_methods(test_images, pathFolder)

            
# Using the special variable
if __name__=="__main__": 
    main()