# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:01:41 2024

@author: Wistan
"""



import numpy as np
import pandas as pd

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity



#------------------     Parameters     ------------------#



# Base paths
pathRoot = './'
pathResults = pathRoot + 'Attrs_absolute/'
pathScores = pathRoot + 'Scores/'


# All methods for files loading
INTERPRET_METHODS = [
                    'BP_captum',
                    'BP_custom',
                    'BP_pair',
                    'Deconv_captum',
                    'Deconv_custom',
                    'EG_captum',
                    'EG_shap',
                    'GradCAM_captum',
                    'GradCAM_pair',
                    'IG0_captum',
                    'IG0_pair',
                     ]


# All groups of methods
GROUPS_METHODS = [
                    'BP',
                    'Deconv',
                    'EG',
                    'GradCAM',
                    'IG0',
                     ]


# Number of available input images
nb_images = 30



#------------------     Utility methods     ------------------#



# Compare two given methods
def compare_methods(method_1, method_2):
    
    # Array of scores
    scores = np.zeros((nb_images, 2))
    
    # For all images
    for i in range (nb_images):
        
        # Load image from method 1 & normalize
        attr_1 = np.load(pathResults + method_1 + '/Raw_attr_' + method_1 + '_Im_' +str(i+1)+ '.npy')
        attr_1 = attr_1 / np.max(attr_1)
        
        # Load image from method 2 & normalize
        attr_2 = np.load(pathResults + method_2 + '/Raw_attr_' + method_2 + '_Im_' +str(i+1)+ '.npy')
        attr_2 = attr_2 / np.max(attr_2)
        
        # Compute MSE & SSIM
        mse = mean_squared_error(attr_1, attr_2)
        ssim = structural_similarity(attr_1, attr_2)
        
        # Store scores
        scores[i] = [mse, ssim]
        
    # Convert to DataFrame & save
    df_scores = pd.DataFrame(data=scores, columns=['MSE', 'SSIM'])
    df_scores.to_csv(pathScores + method_1 + '-' + method_2 + '.csv')
    
    # Return mean & std of scores
    means = np.mean(scores, axis=0)
    stds = np.std(scores, axis=0)
    
    return means, stds



#-------------------     Scores      --------------------#



# Defining main function
def main():
    
    # Array of all means & stds
    df_means_stds = pd.DataFrame(columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'])
    
    
    ### COMPARE METHODS
    
    
    # BP (1)
    means, stds = compare_methods('BP_captum', 'BP_custom')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['BP_captum-BP_custom'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # BP (2)
    means, stds = compare_methods('BP_captum', 'BP_pair')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['BP_captum-BP_pair'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # BP (3)
    means, stds = compare_methods('BP_custom', 'BP_pair')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['BP_custom-BP_pair'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # Deconv
    means, stds = compare_methods('Deconv_captum', 'Deconv_custom')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['Deconv_captum-Deconv_custom'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # EG
    means, stds = compare_methods('EG_captum', 'EG_shap')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['EG_captum-EG_shap'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # GradCAM
    means, stds = compare_methods('GradCAM_captum', 'GradCAM_pair')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['GradCAM_captum-GradCAM_pair'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    # IG0
    means, stds = compare_methods('IG0_captum', 'IG0_pair')
    df_tmp = pd.DataFrame([[means[0], stds[0], means[1], stds[1]]],
                          columns=['MSE_mean', 'MSE_std', 'SSIM_mean', 'SSIM_std'],
                          index=['IG0_captum-IG0_pair'])
    df_means_stds = df_means_stds.append(df_tmp)
    
    
    ### SAVE DATAFRAME
    df_means_stds.to_csv(pathScores + 'Means_Stds.csv')
    
    


# Using the special variable
if __name__=="__main__": 
    main()