# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:45:07 2022

@author: Wistan
"""


import os
import numpy as np
import torch
import pickle


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathOriginalAttr = pathResults + 'Raw_attrs/'
pathAbsoluteAttr = pathResults + 'Raw_attrs (absolute)/'


# All methods for files loading
INTERPRET_METHODS = ['BP',
                      'EG',
                      'IGB',
                      'IGW',
                      'IGBW']


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_263.pt')
# Load params dictionnary
f = open("./params.pkl", "rb")
params = pickle.load(f)
f.close()


# Defining main function
def main():
    
    # For each parameter prediction
    for param_name, param_values in params.items():
            
        print("Param", param_name)
        
        # Range of methods
        for method in (INTERPRET_METHODS):
            
            print('\t Method', method)
            
            # Create full attributions path
            pathLoad = pathOriginalAttr + param_name + '/' + method + '/'
            pathSave = pathAbsoluteAttr + param_name + '/' + method + '/'
            
            # Get all attributions files names
            maps_names = [f for f in os.listdir(pathLoad) if os.path.isfile(os.path.join(pathLoad, f))]
    
            # Range of number of test slices
            for filename in maps_names:
                
                # Load original values results
                original_attr = np.load(pathLoad + filename)
                
                # Convert to absolute values
                absolute_attr = np.abs(original_attr)
                
                # Save absolute values results
                np.save(pathSave + filename, absolute_attr)            


# Using the special variable
if __name__=="__main__": 
    main()