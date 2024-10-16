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
pathResults = pathRoot + 'Results/EG_runs_reprod/'


# Nb of samples
nb_samples = 200
# Images batches size
batch_size = 50
# Nb of runs
nb_runs = [
            2,
            5,
            10,
            15,
          ]


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
    
    # For each run (random seed)
    for rseed in range (np.max(nb_runs)):
        
        print("\t EG Run n°", rseed+1)
        
        # Array for this run's results
        maps_run = []
        
        # For each batch of test_images
        for idxBatch, batch in enumerate(bkgd_batches):
            
            print("\t\t Batch n°", idxBatch)
            
            # GradientExplainer init
            e = shap.GradientExplainer(model, batch)
            
            # Compute SHAP values for given examples
            attrs = e.shap_values(images, nsamples=len(batch), rseed=rseed).squeeze()
            
            # To absolute values
            attrs_abs = np.abs(attrs)
            # Normalize maps
            for j in range(len(attrs_abs)):
                attrs_abs[j] = attrs_abs[j] / np.max(attrs_abs[j])
            
            # Store in run results
            maps_run.append(attrs_abs)
        
        # Average over batches
        maps_run = np.array(maps_run)
        attrs_run = np.mean(maps_run, axis=0)
        
        # Save runs maps as npy
        os.makedirs(pathArch + 'runs/', exist_ok=True)
        np.save(pathArch + 'runs/' + 'Attr_abs_norm_run_' + str(rseed) + '.npy', np.array(attrs_run))
        
        # Store in results array
        all_maps.append(attrs_run)
    
    return np.array(all_maps)



#------------------------------------------------------------------------------#

#----------------------------     AVG / STD     -------------------------------#



def avg_std(all_maps, pathArch, images):
    
    print("\t Avg & Std")
    
    # DataFrame for stats
    df_stats = pd.DataFrame()
    
    # For each run (random seed)
    for runs in nb_runs:
        
        pathRuns = pathArch + str(runs) + '/'
        os.makedirs(pathRuns, exist_ok=True)
        select_runs = all_maps[:runs]
        
        # Swap 1st & 2nd dims
        all_maps_swap = np.swapaxes(select_runs, 0, 1)
        
        # Array of stats per input
        stats_array = np.empty((len(all_maps_swap), 3))
        
        # Create and open stats file
        file = open(pathRuns + 'stats_std_maps.txt', "a")
        file.write('IDX\t\t' + 'MIN\t\t' + 'MAX\t\t' + 'AVG\t\n\n')
        
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
            
            os.makedirs(pathRuns + 'avg_std/', exist_ok=True)
            # Save as npy files
            np.save(pathRuns + 'avg_std/' + 'Avg_idx_' + str(idx) + '.npy', avg_map)
            np.save(pathRuns + 'avg_std/' + 'Std_idx_' + str(idx) + '.npy', std_map)
            
            # Save avg map as image
            P.figure(figsize=(1, 1), dpi=DPI)
            set_size(avg_map.shape[1]*UPSCALE_FACTOR/DPI, (avg_map.shape[0]/DPI)+1)
            ShowImage(images[idx], title='Input Image', ax=P.subplot(ROWS, COLS, 1))
            ShowHeatMap(avg_map, title='Average Map', ax=P.subplot(ROWS, COLS, 2))
            ShowHeatMap(std_map, title='Std Map', ax=P.subplot(ROWS, COLS, 3))
            P.savefig(pathRuns + 'avg_std/' + 'Avg_Std_idx_' + str(idx) + '.tiff')
            P.close()
            
        # Concatenate stats to DataFrame
        df_size = pd.DataFrame(data=stats_array, columns=("Min", "Max", "Avg"))
        df_size["Image"] = np.arange(stop=len(all_maps_swap))
        df_size["Nb_Runs"] = [runs] * len(all_maps_swap)
        df_stats = pd.concat((df_stats, df_size))
            
    # Close stats file
    file.close()
    
    return df_stats



#------------------------------------------------------------------------------#

#-------------------------------     MAIN     ---------------------------------#



def main():
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_2nd.pt').to(device)
    # Split background in batches
    bkgd_batches = torch.split(background, batch_size)
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_30.pt').to(device)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathArch = pathResults + arch + '/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # Call EG experiment function
        all_maps = EG(test_images, bkgd_batches, model, pathArch)
        
        # Avg + Std function
        df_stats = avg_std(all_maps, pathArch, test_images.cpu().detach().squeeze().numpy())
        df_stats.to_csv(pathArch + 'Stats_df.csv')
        
        # Plot stats
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(45, 15), squeeze=False)
        sns.lineplot(x='Image', y='Min', hue='Nb_Runs', data=df_stats, ax=ax[0][0]).set(title='Min Std Value per Input')
        sns.lineplot(x='Image', y='Max', hue='Nb_Runs', data=df_stats, ax=ax[0][1]).set(title='Max Std Value per Input')
        sns.lineplot(x='Image', y='Avg', hue='Nb_Runs', data=df_stats, ax=ax[0][2]).set(title='Avg Std Value per Input')
        fig.tight_layout()
        fig.savefig(pathArch + 'Stats_lineplot.tiff')




# Using the special variable
if __name__=="__main__": 
    main()
