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
loading = './'


# List of folds
trials = [
            0,
            1,
            2,
            3,
            4,
          ]


# All methods to apply
INTERPRET_METHODS = [
                        'BP',
                        'Deconv',
                        'GradCAM',
                        'IG(Min)',
                        'IG(Zero)',
                        'IG(Max)',
                        'IG(Zero-Min)',
                        'IG(Min-Max)',
                        'IG(Zero-Max)',
                        'IG(Avg)',
                        'EG',
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
                        "Min",
                        0.0,
                        "Max",
                      ]


# Fixed Parameters (since training is complete)
params = {
            'net_id': 11,
            'feature': 'CPS',
            'cutoff': 1,
            'batch': 1,
            'size': 192,
            'offsetx': 0,
            'offsety': 0,
            'offsetz': 0,
         }


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#--------------------------         CONCATENATE ALL SCORES AND SAVE         ---------------------------#
        

def concatFidelity(pathMetrics, pathCombined):
    
    print("Concatenation of Fidelity scores")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        # Path of detailed file
        pathSave = pathCombined + trial_folder
        os.makedirs(pathSave, exist_ok=True)
        
        # Empty list for all results
        df_all_results = pd.DataFrame()
        
        # For each perturbation fixed value
        for perturb_value in PERTURBATION_VALUES:
            # Find & extract Fidelity scores
            fidPath = pathMetrics + '/Fidelity_' + str(perturb_value) + '/Graphs/Fidelity_' + str(perturb_value) + '_Avg_t-' + str(trial) + '.csv'
            
            # Load scores dataframe
            df_fid = pd.read_csv(fidPath, index_col=0).fillna(0)
            # Add replace value column
            df_fid['Replace'] = [perturb_value] * len(df_fid.index)
            
            # Append this DataFrame to the overall DataFrame
            df_all_results = pd.concat([df_all_results, df_fid], ignore_index=False)
        
        # Delete undesired results
        df_all_results_sort = df_all_results[df_all_results.index.isin(INTERPRET_METHODS)]
        
        # Save overall results file
        df_all_results_sort.to_csv(pathSave + 'Fidelity_combined_' + str(PERTURBATION_VALUES) + '_t-' + str(trial) + '.csv')


#----------------------------------------------------------------------------------------------#
    
#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_combined(pathCombined):
    
    print("Display Heatmap & Curves")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Path for load & save
        pathSave = pathCombined + trial_folder
        
        # Load global scores DataFrame
        df_all_results = pd.read_csv(pathSave + 'Fidelity_combined_' + str(PERTURBATION_VALUES) + '_t-' + str(trial) + '.csv', index_col=0).fillna(0)
        
        # Init figures
        figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(15, 15))
        figBarplot, axBarplot = plt.subplots(nrows=1, ncols=len(MAP_TYPES), figsize=(12*len(MAP_TYPES), 20), squeeze=False)
        figCurves, axCurves = plt.subplots(nrows=2, ncols=2, figsize=(40, 40), squeeze=False)
        
        # DataFrame for Fidelity scores (mean & std)
        df_mean = pd.DataFrame(index=INTERPRET_METHODS, columns=MAP_TYPES, dtype=float)
        
        # For each map type
        for idxMap, map_type in enumerate(MAP_TYPES):
            
            # Extract map type data
            df_one_mtype = df_all_results[[map_type, 'Replace']]
            
            # Init lists
            fidMean = []
            df_barplot = pd.DataFrame(columns=['Fidelity', 'Method', 'Type'])
            
            # For each method
            for idxMethod, method in enumerate(INTERPRET_METHODS):
                
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
        figHeatmapMean.savefig(pathSave + 'Fidelity_' + str(PERTURBATION_VALUES) + '_t-' + str(trial) + '_Avg_Heatmap.tiff')
        figBarplot.tight_layout()
        figBarplot.savefig(pathSave + 'Fidelity_' + str(PERTURBATION_VALUES) + '_t-' + str(trial) + '_Barplots.tiff')
        figCurves.tight_layout()
        figCurves.savefig(pathSave + 'Fidelity_ReplaceValues_t-' + str(trial) + 'Curves.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # Paths
    pathMetrics = loading + 'Results/Metrics/'
    pathCombined = pathMetrics + 'Fidelity_combined_' + str(PERTURBATION_VALUES) + '/'
    
    # Concatenation of all detailed scores files
    concatFidelity(pathMetrics, pathCombined)
    
    # Fidelity Metric computation and display
    display_Fidelity_combined(pathCombined)

            
# Using the special variable
if __name__=="__main__": 
    main()