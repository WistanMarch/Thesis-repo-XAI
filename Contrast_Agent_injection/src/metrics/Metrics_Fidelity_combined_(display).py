# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# All methods for files loading
INTERPRET_METHODS = [
                        'BP',
                        'Deconv',
                        'IG(0)',
                        'IG(1)',
                        'IG(0-1)',
                        'IGA(01)',
                        'IGA(10)',
                        'EG',
                        'GradCAM',
                        'Random',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
             }


# Fixed values to use for perturbation
PERTURBATION_VALUES = [
                        0.0,
                        0.5,
                        1.0,
                      ]


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#--------------------------         CONCATENATE ALL SCORES AND SAVE         ---------------------------#
        

def concatFidelity(concatFile, pathGraphs):
    
    print("\t Concatenation of Fidelity scores")
    
    # Empty list for all results
    df_all_results = pd.DataFrame()
    
    # Create short version of methods names
    methods_labels = [''.join(c for c in method if c.isupper() or c.isdigit()) for method in INTERPRET_METHODS]
    
    # For each perturbation fixed value
    for perturb_value in PERTURBATION_VALUES:
        # Find & extract Fidelity scores
        fidPath = pathGraphs + '/Fidelity_' + str(perturb_value) + '/Fidelity_' + str(perturb_value) + '_Avg.csv'
        
        # Load scores dataframe
        df_fid = pd.read_csv(fidPath, index_col=0).fillna(0)
        # Add replace value column
        df_fid['Replace'] = [perturb_value] * len(df_fid.index)
        
        # Append this DataFrame to the overall DataFrame
        df_all_results = pd.concat([df_all_results, df_fid], ignore_index=False)
    
    # Delete undesired results
    df_all_results_sort = df_all_results[df_all_results.index.isin(methods_labels)]
    
    # Save overall results file
    df_all_results_sort.to_csv(concatFile)


#----------------------------------------------------------------------------------------------#
    
#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_combined(pathSavePlots, concatFile):
    
    print("\t Display Heatmap & Curves")
    
    # Load global scores DataFrame
    df_all_results = pd.read_csv(concatFile, index_col=0).fillna(0)
    
    # Create short version of methods names
    methods_labels = [''.join(c for c in method if c.isupper() or c.isdigit()) for method in INTERPRET_METHODS]
    
    # Init figures
    figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(15, 15))
    figBarplot, axBarplot = plt.subplots(nrows=1, ncols=len(MAP_TYPES), figsize=(12*len(MAP_TYPES), 20), squeeze=False)
    figCurves, axCurves = plt.subplots(nrows=2, ncols=2, figsize=(40, 40), squeeze=False)
    
    # DataFrame for Fidelity scores (mean & std)
    df_mean = pd.DataFrame(index=methods_labels, columns=MAP_TYPES, dtype=float)
    
    # For each map type
    for idxMap, map_type in enumerate(MAP_TYPES):
        
        # Extract map type data
        df_one_mtype = df_all_results[[map_type, 'Replace']]
        
        # Init lists
        fidMean = []
        df_barplot = pd.DataFrame(columns=['Fidelity', 'Method', 'Type'])
        
        # For each method
        for idxMethod, method in enumerate(methods_labels):
            
            # Extract data for method
            df_one_method = df_one_mtype[df_one_mtype.index == method][map_type]
            
            # Average of the Fidelity scores
            avg_fid = np.mean(df_one_method.to_list())
            fidMean.append(avg_fid)
            df_barplot.loc[len(df_barplot.index)] = [avg_fid, method, map_type]
        
        # Store mean Fidelity scores
        df_mean[map_type] = fidMean
        
        # Plot curves
        sns.lineplot(x='Replace', y=map_type, hue=df_one_mtype.index, data=df_one_mtype, ax=axCurves[idxMap%2][idxMap//2]).set(title='Fidelity '+map_type)
        
        # Plot barplots
        df_mean_copy = df_mean.sort_values(by=map_type, ascending=False)
        sns.barplot(x='Method', y='Fidelity', data=df_barplot, ax=axBarplot[0][idxMap], facecolor="0.5", capsize=.2, order=df_mean_copy.index).set(title='Fidelity barplot: ' + map_type, ylim=(0.0, 1.0))
        axBarplot[0][idxMap].tick_params(axis='x', rotation=90)
    
    # Plot heatmap
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
    
    # Save heatmaps
    figHeatmapMean.savefig(pathSavePlots + 'Fidelity_' + str(PERTURBATION_VALUES) + '_Avg_Heatmap.tiff')
    figBarplot.tight_layout()
    figBarplot.savefig(pathSavePlots + 'Fidelity_' + str(PERTURBATION_VALUES) + '_Barplots.tiff')
    figCurves.tight_layout()
    figCurves.savefig(pathSavePlots + 'Fidelity_ReplaceValues_Curves.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathSaveMetrics = pathRoot + 'Results/' + arch + '/Metrics/'
        pathGraphs = pathSaveMetrics + 'Graphs/'
        pathSavePlots = pathSaveMetrics + 'Graphs/Fidelity_combined_' + str(PERTURBATION_VALUES) + '/'
        os.makedirs(pathSavePlots, exist_ok=True)
        
        # Path + Name for concatenated Fidelity scores file
        concatFile = pathSaveMetrics + 'Detailed/Fidelity/Fidelity_combined_' + str(PERTURBATION_VALUES) + '.csv'
        
        # Concatenation of all detailed scores files
        concatFidelity(concatFile, pathGraphs)
        
        # Fidelity Metric computation and display
        display_Fidelity_combined(pathSavePlots, concatFile)

            
# Using the special variable
if __name__=="__main__": 
    main()