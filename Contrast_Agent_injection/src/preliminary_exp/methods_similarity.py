# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:09:11 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import pandas as pd
from itertools import repeat

from scipy.stats import pearsonr
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity



#--------------------------------     PARAMS     --------------------------------#



# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# List of trained networks
networks = [
            "resnet",
            ]


# All methods for combination
INTERPRET_METHODS = [
                    'BP_custom',
                    'Deconv_captum',
                    'EG_shap',
                    'GradCAM_pair',
                    'IG0_pair',
                    ]


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#--------------------------------------------------------------------------------#

#--------------------------------     UTILS     ---------------------------------#



# Normalization for positive-only maps
def normalize_absolute(image):
    
    image_norm = image / np.max(image)
    return image_norm



# Mean / std + saving DataFrames
def mean_std_save(metric_name, coeffs, XAI_methods, nb_images, pathCorr):
    
    # Save correlation arrays as csv
    array_2D = coeffs.reshape(-1, coeffs.shape[-1])
    # Convert arrays into dataframes
    DF = pd.DataFrame(array_2D, columns=XAI_methods, index=XAI_methods*nb_images)
    DF["Image_idx"] = [x for i in range(nb_images) for x in repeat(i, len(XAI_methods))]
    # Save the dataframe as a csv file
    DF.to_csv(pathCorr + metric_name + '_Coefficients.csv')
    
    # Return mean & std of scores
    mean = np.mean(coeffs, axis=0)
    std = np.std(coeffs, axis=0)
    # Convert arrays into dataframes
    stats = np.concatenate((mean, std), axis=0)
    DF_stats = pd.DataFrame(stats, columns=XAI_methods, index=XAI_methods*2)
    DF_stats["Type"] = [x for i in ("Avg", "Std") for x in repeat(i, len(XAI_methods))]
    # Save the dataframe as a csv file
    DF_stats.to_csv(pathCorr + metric_name + '_Stats.csv')



#--------------------------------------------------------------------------------#

#------------------------     PEARSON + MSE / SSIM     --------------------------#



# Pearson correlation coefficient
def correlation_similarity(pathMaps, pathCorr, images):
    
    # Array for correlation and similarity scores
    pearson_coeffs = np.zeros((len(images), len(INTERPRET_METHODS), len(INTERPRET_METHODS)), dtype=float)
    mse_scores = np.zeros((len(images), len(INTERPRET_METHODS), len(INTERPRET_METHODS)), dtype=float)
    ssim_scores = np.zeros((len(images), len(INTERPRET_METHODS), len(INTERPRET_METHODS)), dtype=float)
    
    # Computing correlation coefficients for each 2-by-2 methods combinations
    for firstIdx, firstMethod in enumerate(INTERPRET_METHODS):
        for secondIdx in range(firstIdx+1, len(INTERPRET_METHODS)):
            
            secondMethod = INTERPRET_METHODS[secondIdx]
            
            print("\t Between", firstMethod, "and", secondMethod)
            
            # Range of number of test slices
            for sliceIdx in range (len(images)):
                
                # Load first method absolute map and normalize
                first_attr = np.load(pathMaps + firstMethod + '/Raw_attr_' + firstMethod + '_Im_' + str(sliceIdx+1) + '.npy')
                first_attr_norm = normalize_absolute(first_attr)
                
                # Load second method absolute map and normalize
                second_attr = np.load(pathMaps + secondMethod + '/Raw_attr_' + secondMethod + '_Im_' + str(sliceIdx+1) + '.npy')
                second_attr_norm = normalize_absolute(second_attr)
                
                # Compute and store Pearson coefficient
                pearson_coeffs[sliceIdx, firstIdx, secondIdx], _ = pearsonr(first_attr_norm.flatten(), second_attr_norm.flatten())
                
                # Compute MSE & SSIM
                mse_scores[sliceIdx, firstIdx, secondIdx] = mean_squared_error(first_attr_norm, second_attr_norm)
                ssim_scores[sliceIdx, firstIdx, secondIdx] = structural_similarity(first_attr_norm, second_attr_norm)
                
                
    # Remove packages from methods names
    XAI_methods = [x.split('_')[0] for x in INTERPRET_METHODS]
    
    # Compute Pearson mean/std + save DataFrames
    mean_std_save("Pearson", pearson_coeffs, XAI_methods, len(images), pathCorr)
    
    # Compute MSE mean/std + save DataFrames
    mean_std_save("MSE", mse_scores, XAI_methods, len(images), pathCorr)
    
    # Compute SSIM mean/std + save DataFrames
    mean_std_save("SSIM", ssim_scores, XAI_methods, len(images), pathCorr)



#--------------------------------------------------------------------------------#

#---------------------------------     MAIN     ---------------------------------#



# if __name__=="__main__":
def main():

    # Images chosen for application of saliency maps
    images = torch.load(pathRoot + 'test_slices_30.pt').numpy()
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # Path for network
        pathArch = pathResults + arch + '/'
        pathMaps = pathArch + 'Raw_attrs(absolute)/'
        pathCorr = pathArch + 'Correlation_Similarity/'
        os.makedirs(pathCorr, exist_ok=True)
        
        # Correlation & Similarity function
        correlation_similarity(pathMaps, pathCorr, images)
    



# Using the special variable
if __name__=="__main__": 
    main()
