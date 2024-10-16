# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:45:07 2022

@author: Wistan
"""


import os
import numpy as np


# Base paths
pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathResults = pathRoot + 'Results/'
pathOriginalAttr = pathResults + 'Raw_attrs/'
pathAbsoluteAttr = pathResults + 'Raw_attrs (absolute)/'


# All methods for files loading
INTERPRET_METHODS = ['Backpropagation',
                    'Deconvolution',
                    'ExpectedGradients',
                    'IntegratedGradients(Black)',
                    'IntegratedGradients(White)',
                    'IntegratedGradients(BlackWhite)']


# Defining main function
def main():

    # Get all subdirectories
    list_all_dirs_files = [x for x in os.walk(pathImages)]
    dirs = list_all_dirs_files[0][1]
    
    # For each subdirectory
    for dirIdx in range(len(dirs)):
        
        dir_path = pathImages + dirs[dirIdx] + '/'
        
        print("Directory :", dir_path)
    
        # Images chosen for application of saliency maps
        images_paths = list_all_dirs_files[dirIdx+1][2]
    
        # Range of number of test slices
        for idx in range (len(images_paths)):
            print('\t Conversion to Absolute Slice number', idx+1)
        
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                # Create full attributions path
                pathLoad = pathOriginalAttr + method + '/' + dirs[dirIdx] + '/'
                pathSave = pathAbsoluteAttr + method + '/' + dirs[dirIdx] + '/'
                
                # Load original values results
                original_attr = np.load(pathLoad + 'Raw_' +method+ '_' +images_paths[idx]+ '.npy')
                
                # Convert to absolute values
                absolute_attr = np.abs(original_attr)
                
                # Save absolute values results
                np.save(pathSave + 'Raw_' +method+ '_' +images_paths[idx]+ '.npy', absolute_attr)            
                
# Using the special variable
if __name__=="__main__": 
    main()