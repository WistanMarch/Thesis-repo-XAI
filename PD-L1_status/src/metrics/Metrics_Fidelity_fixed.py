# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from utils.XAI_utils import XAI_dataset



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
                        "Min",
                        "Max",
                        "InputAvg",
                      ]


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


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


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#-------------------------     FIDELITY SCORES (FIXED VALUE PERTURBATION)     -----------------------#


# Compute Fidelity detailed scores
def Fidelity_fixed(pathResults, perturb_value, pathDetailed):
    
    print("\t Compute MIF/LIF scores")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, net = XAI_dataset(params, trial, device, loading)
        
        # Range of methods
        for method in (INTERPRET_METHODS):
            
            print('\t\t\t Method', method)
                      
            # For each map type
            for map_type in MAP_TYPES:
                
                print("\t\t\t\t Map Type", map_type)
                
                # Path for corresponding attribution maps
                pathLoadFolder = pathResults + MAP_TYPES[map_type] + '/' + method + '/' + trial_folder
                
                # Path for saving detailed scores
                pathSave = pathDetailed + trial_folder
                os.makedirs(pathSave, exist_ok=True)
                
                # Detailed scores array
                detail_scores = []
                
                # For each element in loader
                for step, data in enumerate(loader):
                    
                    # Load original input / Inference
                    in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
                    in_numpy = in_tensor.squeeze().cpu().numpy()
                    pred_in = nn.functional.softmax(net(in_tensor),dim=1).squeeze().detach().cpu().numpy()
                    # Correct or not
                    acc_in = np.argmax(pred_in) == label
                    
                    # Load attribution map
                    ni_map_attr = nib.load(pathLoadFolder + MAP_TYPES[map_type] + '_' + method + '_' + ids[step] + '.nii')
                    map_attr_tensor = torch.from_numpy(ni_map_attr.get_fdata()).to(device)
                    
                    # Find value for filling perturbation image
                    if (perturb_value == 'InputAvg'):
                        fill_perturb = np.mean(in_numpy)
                    elif (perturb_value == 'Min'):
                        fill_perturb = np.min(in_numpy)
                    elif (perturb_value == 'Max'):
                        fill_perturb = np.max(in_numpy)
                    else:
                        fill_perturb = perturb_value
                    
                    # Generate fixed value perturbation image
                    in_perturbed_tensor = torch.full(in_tensor.squeeze().shape, fill_perturb).to(device)
                    
                    
                    # For each percentage
                    for percIdx in range (len(importancePerc)):
                        perc = importancePerc[percIdx]
                        
                        # If 0% then save if correct prediction
                        if (perc == 0):
                            detail_scores.append([acc_in, acc_in, perc, ids[step], map_type, method])
                        else:
                            
                            ################      Compute Most Important First      ################
                    
                            masked_input = torch.clone(torch.from_numpy(in_numpy).to(device))
                            # Hide most salient % of the image
                            mask = map_attr_tensor >= torch.quantile(map_attr_tensor, (100-perc)/100)
                            masked_input[mask] = in_perturbed_tensor[mask]
                            
                            # Compute MIF score
                            with torch.no_grad():
                                output_MIF = nn.functional.softmax(net(masked_input.unsqueeze(0).unsqueeze(0)),dim=1).squeeze().detach().cpu().numpy()
                                y_correct_MIF = np.argmax(output_MIF) == label
                            
                            ################      Compute Least Important First      ################
                    
                            masked_input = torch.clone(torch.from_numpy(in_numpy).to(device))
                            # Hide least salient % of the image
                            mask = map_attr_tensor <= torch.quantile(map_attr_tensor, perc/100)
                            masked_input[mask] = in_perturbed_tensor[mask]
                            
                            # Compute MIF score
                            with torch.no_grad():
                                output_LIF = nn.functional.softmax(net(masked_input.unsqueeze(0).unsqueeze(0)),dim=1).squeeze().detach().cpu().numpy()
                                y_correct_LIF = np.argmax(output_LIF) == label
                            
                            #########################################################################
                            
                            # Save in detailed array
                            detail_scores.append([y_correct_MIF, y_correct_LIF, perc, ids[step], map_type, method])
                
                # Convert into dataframe & save
                DF_detail_scores = pd.DataFrame(detail_scores, columns=['MIF', 'LIF', 'Percentage', 'ID', 'Map', 'Method'])
                DF_detail_scores.to_csv(pathSave + 'detailed_' + str(perturb_value) + '_' + method + '_' + map_type + '.csv')
                

#----------------------------------------------------------------------------------------------------#

#-------------------------         CONCATENATE ALL SCORES AND SAVE         --------------------------#
        

def concatDetailed(pathDetailed):
    
    print("\t Concatenation of all Results")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        # Path of detailed files
        pathScores = pathDetailed + trial_folder
        
        # Empty list for all results
        df_all_results = pd.DataFrame()
        
        # Get detailed scores filenames
        detailed_scores_files = [f for f in os.listdir(pathScores) if os.path.isfile(os.path.join(pathScores, f))]
        
        # For each file in directory
        for name in detailed_scores_files:
        
            # Load scores dataframe
            df_scores = pd.read_csv(pathScores + name, index_col=0).fillna(0)
        
            # Append this DataFrame to the overall DataFrame
            df_all_results = pd.concat([df_all_results, df_scores], ignore_index=True)
        
        df_all_results_sort = df_all_results[df_all_results['Method'].isin(INTERPRET_METHODS)]
        
        # Save overall results file
        df_all_results_sort.to_csv(pathDetailed + 'all_fidelity_scores_t-' + str(trial) + '.csv')
        

#-----------------------------------------------------------------------------------------------------#
    
#--------------------------         FIDELITY DISPLAY (FIXED VALUE)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_fixed(pathDetailed, pathGraphs, perturb_value):
    
    print("\t Display Heatmap & Curves")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t\t Trial', trial)
        
        # Create dataset for XAI
        ids, _, _, _ = XAI_dataset(params, trial, device, loading)
        
        # Load global scores DataFrame
        df_all_results = pd.read_csv(pathDetailed + 'all_fidelity_scores_t-' + str(trial) + '.csv', index_col=0).fillna(0)
        
        # Init heatmap figure
        figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(15, 15))
        # Init MIF/LIF & Barplots figures
        figMIFLIFCurves, axMIFLIFCurves = plt.subplots(nrows=len(INTERPRET_METHODS), ncols=len(MAP_TYPES), figsize=(20*len(MAP_TYPES), 10*len(INTERPRET_METHODS)), squeeze=False)
        figBarplot, axBarplot = plt.subplots(nrows=1, ncols=len(MAP_TYPES), figsize=(12*len(MAP_TYPES), 20), squeeze=False)
        
        # DataFrame for Fidelity scores (mean & std)
        df_mean = pd.DataFrame(index=INTERPRET_METHODS, columns=MAP_TYPES, dtype=float)
        
        # For each map type
        for idxMap, map_type in enumerate(MAP_TYPES):
            
            # Extract map type data
            df_one_mtype = df_all_results[df_all_results['Map'] == map_type]
            
            # Init lists
            fidMeanMap = []
            fidScoresCurves = []
            
            # For each method
            for idxMethod, method in enumerate(INTERPRET_METHODS):
                
                # DataFrame of all outputs
                df_preds = pd.DataFrame()
                
                # Extract data for method
                df_one_method = df_one_mtype[df_one_mtype['Method'] == method]
                
                # Extract MIF method data
                MIF_array_method = df_one_method[['MIF']].to_numpy().reshape(len(ids), len(importancePerc))
                MIF_accuracy = np.sum(MIF_array_method, axis=0) / len(ids)
                
                # Extract LIF method data
                LIF_array_method = df_one_method[['LIF']].to_numpy().reshape(len(ids), len(importancePerc))
                LIF_accuracy = np.sum(LIF_array_method, axis=0) / len(ids)
                
                # AUCs & Fidelity score
                AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_accuracy)
                AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_accuracy)
                Fid_score = AUC_LIF - AUC_MIF
                fidScoresCurves.append([Fid_score, method])
                
                # For plotting the MIF & LIF curves
                df_preds['Accuracy'] = list(LIF_accuracy) + list(MIF_accuracy)
                df_preds['Perc'] = 2 * importancePerc
                df_preds['MIF_LIF'] = ['LIF'] * len(importancePerc) + ['MIF'] * len(importancePerc)
                
                # Plot MIF & LIF curves
                sns.lineplot(x='Perc', y='Accuracy', hue='MIF_LIF', data=df_preds, ax=axMIFLIFCurves[idxMethod][idxMap]).set(title='MIF & LIF Curves: ' + map_type + ' Method ' + method, ylim=(0.0, 1.0))
            
                # Mean & Std of Fidelity scores
                fidMeanMap.append(Fid_score)
                
            # Save scores for this map type
            df_mean[map_type] = fidMeanMap
            df_fid = pd.DataFrame(data=fidScoresCurves, columns=['Fidelity', 'Method'])
            df_fid['Map'] = [map_type] * len(fidScoresCurves)
            
            # Plot barplots
            x_lim = (-0.4, 0.601)
            df_mean_copy = df_mean.sort_values(by=map_type, ascending=False)
            sns.barplot(data=df_fid, x='Fidelity', y='Method', hue='Map', ax=axBarplot[0][idxMap], gap=.1, order=df_mean_copy.index).set(xlim=x_lim)
            # Add title if needed: "title='Fidelity barplot: ' + map_type, "
            axBarplot[0][idxMap].set_xticks(np.arange(x_lim[0], x_lim[1], 0.1))
    
        # Fill heatmaps
        sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
        axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
        # Save heatmaps
        figHeatmapMean.savefig(pathGraphs + 'Fidelity_' + str(perturb_value) + '_Avg_Heatmap_t-' + str(trial) + '.tiff')
        df_mean.to_csv(pathGraphs + 'Fidelity_' + str(perturb_value) + '_Avg_t-' + str(trial) + '.csv')
        # Save curves
        figMIFLIFCurves.tight_layout()
        figMIFLIFCurves.savefig(pathGraphs + 'Fidelity_' + str(perturb_value) + '_MIFLIF_Curves_t-' + str(trial) + '.tiff')
        figBarplot.tight_layout()
        figBarplot.savefig(pathGraphs + 'Fidelity_' + str(perturb_value) + '_Barplots_t-' + str(trial) + '.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each perturbation fixed value
    for perturb_value in PERTURBATION_VALUES:
        
        print("Perturbation Value " + str(perturb_value))
        
        # Paths
        pathResults = loading + 'Results/'   
        pathFid = pathResults + 'Metrics/Fidelity_' + str(perturb_value) + '/'
        pathGraphs = pathFid + 'Graphs/'
        os.makedirs(pathGraphs, exist_ok=True)
        pathDetailed = pathFid + 'Detailed/'
        
        # # Fidelity MIF / LIF computation
        # Fidelity_fixed(pathResults, perturb_value, pathDetailed)
        
        # # Concatenation of all detailed scores files
        # concatDetailed(pathDetailed)
        
        # Fidelity Metric computation and display
        display_Fidelity_fixed(pathDetailed, pathGraphs, perturb_value)


            
# Using the special variable
if __name__=="__main__": 
    main()