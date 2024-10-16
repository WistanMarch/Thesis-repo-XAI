# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


from utils.XAI_utils import XAI_dataset
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



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
                        'EG',
                        'IG(Zero)',
                        'IG(Min)',
                        'IG(Max)',
                        'IG(Zero-Min)',
                        'IG(Min-Max)',
                        'IG(Zero-Max)',
                        'IG(Avg)',
                        'Random',
                        # 'SG+BP',
                        # 'SG+Deconv',
                        # 'SG+GradCAM',
                        # 'SG+IG(Zero)',
                        # 'SG+EG',
                        # 'SGSQ+BP',
                        # 'SGSQ+Deconv',
                        # 'SGSQ+GradCAM',
                        # 'SGSQ+IG(Zero)',
                        # 'SGSQ+EG',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
             }


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#---------------------------------     FIDELITY SCORES (BLUR PERTURBATION)     -------------------------------#


# Compute Fidelity detailed scores
def Fidelity_blur(pathResults, pathDetailed):
    
    print("Compute MIF/LIF scores")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, net = XAI_dataset(params, trial, device, loading)
        
        # Path for perturbed images
        pathPerturb = loading + 'dataset_crop-x-0-y-0-z-0/data_perturbed/' + trial_folder
            
        # For each method
        for method in (INTERPRET_METHODS):
            
            print('\t\t Method', method)
                      
            # For each map type
            for map_type in MAP_TYPES:
                    
                print("\t\t\t Map Type", map_type)
            
                # Path for corresponding attribution maps
                pathLoadFolder = pathResults + MAP_TYPES[map_type] + '/' + method + '/' + trial_folder
                
                # Path for saving detailed scores
                pathSave = pathDetailed + trial_folder
                os.makedirs(pathSave, exist_ok=True)
                
                # Detailed scores array
                detail_scores = []
                
                # For each element in loader
                for step, data in enumerate(loader):
                    
                    # # Display id
                    # print("\t\t\t\t ID", ids[step])
                    
                    # Load perturbed image
                    ni_img_perturbed = nib.load(pathPerturb + 'blur_' + ids[step] + '.nii')
                    in_perturbed_tensor = torch.from_numpy(ni_img_perturbed.get_fdata()).to(device).float()
                    
                    # Load original input / Inference
                    in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
                    in_numpy = in_tensor.squeeze().cpu().numpy()
                    pred_in = nn.functional.softmax(net(in_tensor),dim=1).squeeze().detach().cpu().numpy()
                    
                    # Load attribution map
                    ni_map_attr = nib.load(pathLoadFolder + MAP_TYPES[map_type] + '_' + method + '_' + ids[step] + '.nii')
                    map_attr_tensor = torch.from_numpy(ni_map_attr.get_fdata()).to(device)
                    
                    # For each percentage
                    for percIdx in range (len(importancePerc)):
                        perc = importancePerc[percIdx]
                        
                        # If 0% then save original prediction
                        if (perc == 0):
                            detail_scores.append([pred_in[label], pred_in[label], perc, ids[step], map_type, method])
                        else:
                            
                            ################      Compute Most Important First      ################
                    
                            masked_input = torch.clone(torch.from_numpy(in_numpy).to(device))
                            # Hide most salient % of the image
                            mask = map_attr_tensor >= torch.quantile(map_attr_tensor, (100-perc)/100)
                            masked_input[mask] = in_perturbed_tensor[mask]
                            
                            # Compute MIF score
                            with torch.no_grad():
                                output_MIF = nn.functional.softmax(net(masked_input.unsqueeze(0).unsqueeze(0)),dim=1).squeeze().detach().cpu().numpy()[label]
                            
                            ################      Compute Least Important First      ################
                    
                            masked_input = torch.clone(torch.from_numpy(in_numpy).to(device))
                            # Hide least salient % of the image
                            mask = map_attr_tensor <= torch.quantile(map_attr_tensor, perc/100)
                            masked_input[mask] = in_perturbed_tensor[mask]
                            
                            # Compute MIF score
                            with torch.no_grad():
                                output_LIF = nn.functional.softmax(net(masked_input.unsqueeze(0).unsqueeze(0)),dim=1).squeeze().detach().cpu().numpy()[label]
                            
                            #########################################################################
                        
                            # Save in detailed array
                            detail_scores.append([output_MIF, output_LIF, perc, ids[step], map_type, method])
         
                # Convert into dataframe & save
                DF_detail_scores = pd.DataFrame(detail_scores, columns=['MIF', 'LIF', 'Percentage', 'ID', 'Map', 'Method'])
                DF_detail_scores.to_csv(pathSave + 'detailed_blur_' + method + '_' + map_type + '.csv')
                

#------------------------------------------------------------------------------------------------------#

#--------------------------         CONCATENATE ALL SCORES AND SAVE         ---------------------------#


def concatDetailed(pathDetailed):
    
    print("Concatenation of all Results")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
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
        
        # Keep only data for selected methods
        df_all_results_sort = df_all_results[df_all_results['Method'].isin(INTERPRET_METHODS)]
        
        # Save overall results file
        df_all_results_sort.to_csv(pathDetailed + 'all_fidelity_scores_t-' + str(trial) + '.csv')


#----------------------------------------------------------------------------------------------#

#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_blur(pathDetailed, pathGraphs):
    
    print("Display Heatmap & Curves")
    
    os.makedirs(pathGraphs, exist_ok=True)
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Create dataset for XAI
        ids, _, _, _ = XAI_dataset(params, trial, device, loading)
        
        # Load global scores DataFrame
        df_all_results = pd.read_csv(pathDetailed + 'all_fidelity_scores_t-' + str(trial) + '.csv', index_col=0).fillna(0)
        
        # Init heatmap figure
        figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(15, 15))
        figHeatmapStd, axHeatmapStd = plt.subplots(figsize=(15, 15))
        # Init MIF/LIF & Barplots figures
        figMIFLIFCurves, axMIFLIFCurves = plt.subplots(nrows=len(INTERPRET_METHODS), ncols=len(MAP_TYPES), figsize=(20*len(MAP_TYPES), 10*len(INTERPRET_METHODS)), squeeze=False)
        figBarplot, axBarplot = plt.subplots(nrows=1, ncols=len(MAP_TYPES), figsize=(12*len(MAP_TYPES), 20), squeeze=False)
        figBarAll, axBarAll = plt.subplots(figsize=(20, 1*len(INTERPRET_METHODS)*len(MAP_TYPES)), squeeze=False)
        
        # DataFrame for Fidelity scores (mean & std)
        df_mean = pd.DataFrame(index=INTERPRET_METHODS, columns=MAP_TYPES, dtype=float)
        df_std = pd.DataFrame(index=INTERPRET_METHODS, columns=MAP_TYPES, dtype=float)
        # DataFrame for boxplots
        df_bars = pd.DataFrame()
            
        # For each map type
        for idxMap, map_type in enumerate(MAP_TYPES):
            
            # Extract map type data
            df_one_mtype = df_all_results[df_all_results['Map'] == map_type]
            
            # Init lists
            fidMeanMap = []
            fidStdMap = []
            fidScoresCurves = []
            
            # For each method
            for idxMethod, method in enumerate(INTERPRET_METHODS):
                
                # List of Fidelity scores per slice
                fidScores = []
                # DataFrame of all outputs
                df_preds = pd.DataFrame()
                
                # Extract data for method
                df_one_method = df_one_mtype[df_one_mtype['Method'] == method]
                
                # For each element in loader
                for id_input in ids:
                    
                    # Extract data for slice
                    df_one_input = df_one_method[df_one_method['ID'] == int(id_input)]
            
                    # Extract MIF & LIF results
                    MIF_res = list(df_one_input[['MIF']].to_numpy())
                    LIF_res = list(df_one_input[['LIF']].to_numpy())
                    # Compute area between MIF & LIF curves
                    AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
                    AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
                    Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
        
                    # Save output scores (LIF then MIF)
                    df_preds_tmp = pd.DataFrame(data=LIF_res+MIF_res, columns=['Output'])
                    df_preds_tmp['Perc'] = 2 * importancePerc
                    df_preds_tmp['MIF_LIF'] = ['LIF'] * len(importancePerc) + ['MIF'] * len(importancePerc)
                    df_preds_tmp['Slice'] = [id_input] * len(importancePerc) * 2
                    df_preds = pd.concat([df_preds, df_preds_tmp], ignore_index=True)
                    fidScores.append(Fid_score)
                    fidScoresCurves.append([Fid_score, AUC_LIF, AUC_MIF, id_input, method])
                    
                # Plot MIF & LIF curves
                sns.lineplot(x='Perc', y='Output', hue='MIF_LIF', data=df_preds, ax=axMIFLIFCurves[idxMethod][idxMap]).set(title='MIF & LIF Curves: ' + map_type + ' Method ' + method, ylim=(0.0, 1.0))
            
                # Mean & Std of Fidelity scores
                fidMeanMap.append(np.mean(fidScores))
                fidStdMap.append(np.std(fidScores))
                        
            # Save scores for this map type
            df_mean[map_type] = fidMeanMap
            df_std[map_type] = fidStdMap
            df_fid = pd.DataFrame(data=fidScoresCurves, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'Slice', 'Method'])
            df_fid['Map'] = [map_type] * len(fidScoresCurves)
            df_bars = pd.concat([df_bars, df_fid], ignore_index=True)
            df_fid.to_csv(pathGraphs + 'Fidelity_per_input_' + map_type + '_t-' + str(trial) + '.csv')
            
            # Plot barplots
            df_mean_copy = df_mean.sort_values(by=map_type, ascending=False)
            sns.barplot(data=df_fid, x='Fidelity', y='Method', hue='Map', ax=axBarplot[0][idxMap], gap=.1, errorbar='sd', order=df_mean_copy.index).set(xlim=(0.3, 1.0))
            # Add title if needed: "title='Fidelity barplot: ' + map_type, "
        
        # Create overall barplot
        df_mean_order = df_mean.sort_values(by=next(iter(MAP_TYPES)), ascending=False)
        sns.barplot(data=df_bars, x='Fidelity', y='Method', hue='Map', gap=.1, ax=axBarAll[0,0], errorbar='sd', order=df_mean_order.index).set(xlim=(0.2, 1.0))
        # Create heatmaps
        sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
        sns.heatmap(df_std, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapStd)
        axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
        axHeatmapStd.set_yticklabels(df_std.index.values, rotation=0, rotation_mode='anchor', ha='right')
        
        # Save overall barplot
        figBarAll.tight_layout()
        figBarAll.savefig(pathGraphs + 'Fidelity_Barplot_combined_t-' + str(trial) + '.tiff')
        # Save heatmaps
        figHeatmapMean.savefig(pathGraphs + 'Fidelity_Avg_Heatmap_t-' + str(trial) + '.tiff')
        figHeatmapStd.savefig(pathGraphs + 'Fidelity_Std_Heatmap_t-' + str(trial) + '.tiff')
        df_mean.to_csv(pathGraphs + 'Fidelity_Avg_t-' + str(trial) + '.csv')
        df_std.to_csv(pathGraphs + 'Fidelity_Std_t-' + str(trial) + '.csv')
        # Save curves
        figMIFLIFCurves.tight_layout()
        figMIFLIFCurves.savefig(pathGraphs + 'Fidelity_MIFLIF_Curves_t-' + str(trial) + '.tiff')
        figBarplot.tight_layout()
        figBarplot.savefig(pathGraphs + 'Fidelity_Barplots_t-' + str(trial) + '.tiff')


#----------------------------------------------------------------------------------------------#

#---------------------         FIDELITY DISPLAY (BLUR) PER METHOD         ---------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_blur_per_method(pathDetailed, pathGraphs):
    
    print("Display Heatmap & Curves")
    
    os.makedirs(pathGraphs, exist_ok=True)
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Create dataset for XAI
        ids, _, _, _ = XAI_dataset(params, trial, device, loading)
        
        # Load global scores DataFrame
        df_all_results = pd.read_csv(pathDetailed + 'all_fidelity_scores_t-' + str(trial) + '.csv', index_col=0).fillna(0)
        
        # Init Barplots figure
        figBarAll, axBarAll = plt.subplots(nrows=1, ncols=len(MAP_TYPES), figsize=(15*len(MAP_TYPES), 1.2*len(INTERPRET_METHODS)), squeeze=False)
        
        # DataFrame for Fidelity scores (mean & std)
        df_mean = pd.DataFrame(index=INTERPRET_METHODS, columns=MAP_TYPES, dtype=float)
        # DataFrame for boxplots
        df_bars = pd.DataFrame()
            
        # For each map type
        for idxMap, map_type in enumerate(MAP_TYPES):
            
            # Extract map type data
            df_one_mtype = df_all_results[df_all_results['Map'] == map_type]
            
            # Init lists
            fidMeanMap = []
            fidStdMap = []
            fidScoresCurves = []
            
            # For each method
            for idxMethod, method in enumerate(INTERPRET_METHODS):
                
                # List of Fidelity scores per slice
                fidScores = []
                # DataFrame of all outputs
                df_preds = pd.DataFrame()
                
                # Extract data for method
                df_one_method = df_one_mtype[df_one_mtype['Method'] == method]
                
                # Extract base method
                base_method = method.replace('SGSQ+', '').replace('SG+', '').replace('ImgReg+', '')
                
                # For each element in loader
                for id_input in ids:
                    
                    # Extract data for slice
                    df_one_input = df_one_method[df_one_method['ID'] == int(id_input)]
            
                    # Extract MIF & LIF results
                    MIF_res = list(df_one_input[['MIF']].to_numpy())
                    LIF_res = list(df_one_input[['LIF']].to_numpy())
                    # Compute area between MIF & LIF curves
                    AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
                    AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
                    Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
        
                    # Save output scores (LIF then MIF)
                    df_preds_tmp = pd.DataFrame(data=LIF_res+MIF_res, columns=['Output'])
                    df_preds_tmp['Perc'] = 2 * importancePerc
                    df_preds_tmp['MIF_LIF'] = ['LIF'] * len(importancePerc) + ['MIF'] * len(importancePerc)
                    df_preds_tmp['Slice'] = [id_input] * len(importancePerc) * 2
                    df_preds = pd.concat([df_preds, df_preds_tmp], ignore_index=True)
                    fidScores.append(Fid_score)
                    fidScoresCurves.append([Fid_score, AUC_LIF, AUC_MIF, id_input, method, base_method])
                
                # Mean & Std of Fidelity scores
                fidMeanMap.append(np.mean(fidScores))
                fidStdMap.append(np.std(fidScores))
            
            # Save scores for this map type
            df_mean[map_type] = fidMeanMap
            df_fid = pd.DataFrame(data=fidScoresCurves, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'Slice', 'Method', 'Base_Method'])
            df_fid['Map'] = [map_type] * len(fidScoresCurves)
            df_bars = pd.concat([df_bars, df_fid], ignore_index=True)
        
            # Create overall barplot
            df_mean_order = df_mean.sort_values(by=next(iter(MAP_TYPES)), ascending=False)
            sns.barplot(data=df_bars, x='Fidelity', y='Method', hue='Base_Method', gap=.1, ax=axBarAll[0,idxMap], errorbar='sd', order=df_mean_order.index).set(xlim=(0.35, 1.0))
            
        # Save overall barplot
        figBarAll.tight_layout()
        figBarAll.savefig(pathGraphs + 'Fidelity_Barplot_per_method_t-' + str(trial) + '.tiff')


#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # Paths
    pathResults = loading + 'Results/'   
    pathFid = pathResults + 'Metrics/Fidelity_Blur/'
    pathGraphs = pathFid + 'Graphs/'
    pathDetailed = pathFid + 'Detailed/'
    
    # # For Fidelity with a blurred version of input
    # Fidelity_blur(pathResults, pathDetailed)
    
    # # Concatenation of all detailed scores files
    # concatDetailed(pathDetailed)
    
    # Fidelity Metric computation and display
    display_Fidelity_blur(pathDetailed, pathGraphs)
    
    # # Fidelity Metric computation and display
    # display_Fidelity_blur_per_method(pathDetailed, pathGraphs)

            
# Using the special variable
if __name__=="__main__": 
    main()