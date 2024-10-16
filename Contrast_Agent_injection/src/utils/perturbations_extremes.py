# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:40:15 2023

@author: Wistan
"""


"""
From all valid perturbations (i.e with model output below the uncertainty limit),
extract the Min & Max of a specified column (e.g. MSE or MSE/Range)
"""



import numpy as np
import pandas as pd



#------------------     Parameters     ------------------#



# Base paths
pathRoot = './'


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# Uncertainty limit (for perturbation sorting) above or below 1/nb_classes
uncertain_limit = 0.1


# Path of perturbations list
input_DF = pathRoot + 'Results/Blur_Perturbation/'


# Name of column to extract
column_name = 'MSE/Range'



#------------------     Min / Max Extraction     ------------------#



# For each network
for arch in networks:
        
    print("\t Network " + arch)
    
    # Perturbations path for network
    pathDF = input_DF + arch + '/'
    
    # Extract all perturbations
    df_all_results = pd.read_csv(pathDF + 'All_Perturbations.csv', index_col=0).fillna(0)
    
    # Min & Max for this network
    min_max_net = []
    
    # List of IDs
    list_ID = list(np.unique(df_all_results['ID'].to_list()))
    
    # For each ID in file
    for ID in list_ID:
        
        # ID perturbations
        df_im = df_all_results[df_all_results['ID'] == ID]
        # Valid perturbations
        df_diff = df_im[df_im['Uncert_score'] <= uncertain_limit]
        # Data of specified column
        data_list = list(np.unique(df_diff[column_name].to_list()))
        
        # Get min & max
        min_max_net.append([np.min(data_list), np.max(data_list)])
    
    # Save min & max
    min_max_net = np.array(min_max_net)

    # Display
    print(np.min(min_max_net), np.max(min_max_net))