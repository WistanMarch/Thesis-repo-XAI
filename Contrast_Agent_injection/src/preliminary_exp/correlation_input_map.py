# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns



#--------------------------------     PARAMS     --------------------------------#



# List of trained networks
networks = [
            "resnet",
            # "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# All methods for files loading
INTERPRET_METHODS = [
                    'Backpropagation',
                    'Deconvolution',
                    'IntegratedGradients(Black)',
                    'ExpectedGradients',
                    'GradCAM',
                     ]


# Use raw or absolute values
use_abs = [
                False,
                True
             ]


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_30.pt')


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#--------------------------------------------------------------------------------#

#--------------------------------     UTILS     ---------------------------------#



# Normalization for positive-only maps
def normalize_absolute(image):
    
    image_norm = image / np.max(image)
    return image_norm



# Standardization for bipolar maps (negative & positive values)
def standardize_raw(image):
    
    extreme = np.max(np.abs(image))
    image_stand = image / extreme
    return image_stand



#--------------------------------------------------------------------------------#

#---------------------------------     MAIN     ---------------------------------#



# Defining main function
def main():

    # For each network
    for arch in networks:
        
        print("Network " + arch)
        
        pathArch = pathResults + arch + '/Raw_attrs/'
        
        # For each map type
        for absolute in use_abs:
            
            print("\t Absolute " + str(absolute))
            
            # Change paths if absolute values
            if (absolute):
                pathAbs = pathArch[:-1] + '(absolute)' + pathArch[-1]
                process_fn = normalize_absolute
            else:
                pathAbs = pathArch
                process_fn = standardize_raw
            
            # Range of number of test slices
            for idx, im in enumerate(test_images):
                
                # Init scatterplot figure
                figIm, axIm = plt.subplots(nrows=1, ncols=len(INTERPRET_METHODS), figsize=(15*len(INTERPRET_METHODS), 15), squeeze=False)
                
                # Convert to numpy and flatten
                im_np = im.numpy()
                im_np_flat = im_np.flatten()
                
                # Range of methods
                for mIdx, method in enumerate(INTERPRET_METHODS):
                    
                    # Create full attributions path
                    pathLoad = pathAbs + method + '/'
                    
                    # Load raw and XRAI results
                    attr = np.load(pathLoad + 'Raw_attr_' + method + '_Im_' + str(idx+1) + '.npy')
                    
                    # Normalize / standardize then flatten map
                    attr_process = process_fn(attr)
                    attr_flat = attr_process.flatten()
                    
                    # Create DataFrame
                    df_im = pd.DataFrame(data=np.transpose([im_np_flat, attr_flat]), columns=('Input', 'Map'))
                    
                    # Plot data
                    y_lim = (0.0, 1.0) if absolute else (-1.0, 1.0)
                    sns.scatterplot(data=df_im, x='Input', y='Map', ax=axIm[0][mIdx]).set(title='Method ' + method, ylim=y_lim)
                
                # Setup saving path
                pathOutput = pathResults + 'Input_vs_Map/' + arch + '/Absolute_' + str(absolute) + '/'
                os.makedirs(pathOutput, exist_ok=True)
                
                # Save figure and clean
                figIm.tight_layout()
                figIm.savefig(pathOutput + 'Im_' + str(idx) + '.tiff')
                figIm.clf()
            




# Using the special variable
if __name__=="__main__": 
    main()