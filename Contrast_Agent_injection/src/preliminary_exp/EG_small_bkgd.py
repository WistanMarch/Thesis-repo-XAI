# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:40:31 2021

@author: wista
"""


import os
import torch
import numpy as np
import pandas as pd
import pickle
import shap

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pylab as P

from utils_display import set_size, ShowHeatMap, ShowImage



#-----------------------------     PARAMETERS     -----------------------------#



# List of trained networks
networks = [
            "resnet",
            # "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/EG_bkgd_reprod/'


# Nb of samples
nb_samples = 200
# Same seed for all batches
rseed = 42
# Images batches size
batch_size = [
                10,
                20,
                40,
             ]
# Number of batches (limited by highest batch_size value)
nb_batches = nb_samples / np.max(batch_size)


# Set up matplot lib figures.
ROWS = 1
COLS = 3
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#------------------------------------------------------------------------------#

#-------------------------------     UTILS     --------------------------------#



# Write & save txt file with stats
def write_stats(file, idx, min_stat, max_stat, avg_stat):
    
    # Original pixel values
    file.write(str(idx) + '\t\t' + '%.4f'%min_stat + '\t\t' + '%.4f'%max_stat + '\t\t' + '%.4f'%avg_stat + '\t\n')
    
    return file



#------------------------------------------------------------------------------#

#---------------------------------     EG     ---------------------------------#



def EG(images, bkgd_batches, model, pathArch):
    
    # Array of all maps
    all_maps = []
    
    # For each batch of test_images
    for idx, batch in enumerate(bkgd_batches):
        
        if (idx < nb_batches):
        
            print("\t\t Batch n°", idx)
            
            # GradientExplainer init
            e = shap.GradientExplainer(model, batch)
            # Compute SHAP values for given examples
            attrs = e.shap_values(images, nsamples=nb_samples, rseed=rseed).squeeze()
            
            # To absolute values
            attrs_abs = np.abs(attrs)
            # Normalize maps
            for j in range(len(attrs_abs)):
                attrs_abs[j] = attrs_abs[j] / np.max(attrs_abs[j])
            
            # Save batch maps as npy
            os.makedirs(pathArch + 'batches/', exist_ok=True)
            np.save(pathArch + 'batches/' + 'Attr_abs_norm_batch_' + str(idx) + '.npy', attrs_abs)
            
            # Store in results array
            all_maps.append(attrs_abs)
    
    return np.array(all_maps)



#------------------------------------------------------------------------------#

#----------------------------     AVG / STD     -------------------------------#



def avg_std(all_maps, pathArch, images):
    
    print("\t\t Avg & Std")
    
    # Swap 1st & 2nd dims
    all_maps_swap = np.swapaxes(all_maps, 0, 1)
    
    # Create and open stats file
    file = open(pathArch + 'stats_std_maps.txt', "a")
    file.write('IDX\t\t' + 'MIN\t\t' + 'MAX\t\t' + 'AVG\t\n\n')
    
    # Array of stats per input
    stats_array = np.empty((len(images), 3))
    
    # For each input image
    for idx, data in enumerate(all_maps_swap):
        
        # Avg & std map
        avg_map = np.mean(data, axis=0)
        std_map = np.std(data, axis=0)
        
        # Max / min / avg of std map + store
        min_std_map = np.min(std_map)
        max_std_map = np.max(std_map)
        avg_std_map = np.mean(std_map)
        stats_array[idx] = (min_std_map, max_std_map, avg_std_map)
        
        # Save std map stats
        file = write_stats(file, idx, min_std_map, max_std_map, avg_std_map)
        
        os.makedirs(pathArch + 'avg_std/', exist_ok=True)
        # Save as npy files
        np.save(pathArch + 'avg_std/' + 'Avg_idx_' + str(idx) + '.npy', avg_map)
        np.save(pathArch + 'avg_std/' + 'Std_idx_' + str(idx) + '.npy', std_map)
        
        # Save avg map as image
        P.figure(figsize=(1, 1), dpi=DPI)
        set_size(avg_map.shape[1]*UPSCALE_FACTOR/DPI, (avg_map.shape[0]/DPI)+1)
        ShowImage(images[idx], title='Input Image', ax=P.subplot(ROWS, COLS, 1))
        ShowHeatMap(avg_map, title='Average Map', ax=P.subplot(ROWS, COLS, 2))
        ShowHeatMap(std_map, title='Std Map', ax=P.subplot(ROWS, COLS, 3))
        P.savefig(pathArch + 'avg_std/' + 'Avg_Std_idx_' + str(idx) + '.tiff')
        P.close()
    
    # Close stats file
    file.close()
    
    return stats_array



#------------------------------------------------------------------------------#

#-------------------------------     MAIN     ---------------------------------#



def main():
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_2nd.pt').to(device)
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_30.pt').to(device)
    
    # For each network
    for arch in networks:
        
        print("Network " + arch)
        
        pathArch = pathResults + arch + '/'
        
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # DataFrame for stats
        df_stats = pd.DataFrame()
        
        # For each size of background group
        for size in batch_size:
            
            print("\t Group size " + str(size))
            
            # Split background in 20-images batches
            bkgd_batches = torch.split(background, size)
            
            # Update path
            pathSize = pathArch + str(size) + '/'
            
            # Call EG experiment function
            all_maps = EG(test_images, bkgd_batches, model, pathSize)
            
            # Avg + Std function
            stats_size = avg_std(all_maps, pathSize, test_images.cpu().detach().squeeze().numpy())
            df_size = pd.DataFrame(data=stats_size, columns=("Min", "Max", "Avg"))
            df_size["Image"] = np.arange(stop=len(test_images))
            df_size["Batch_size"] = [size] * len(test_images)
            
            df_stats = pd.concat((df_stats, df_size))
            
        # Plot stats
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(45, 15), squeeze=False)
        sns.lineplot(x='Image', y='Min', hue='Batch_size', data=df_stats, ax=ax[0][0]).set(title='Min Std Value per Input')
        sns.lineplot(x='Image', y='Max', hue='Batch_size', data=df_stats, ax=ax[0][1]).set(title='Max Std Value per Input')
        sns.lineplot(x='Image', y='Avg', hue='Batch_size', data=df_stats, ax=ax[0][2]).set(title='Avg Std Value per Input')
        fig.tight_layout()
        fig.savefig(pathArch + 'Stats_lineplot.tiff')
        




# Using the special variable
if __name__=="__main__": 
    main()
