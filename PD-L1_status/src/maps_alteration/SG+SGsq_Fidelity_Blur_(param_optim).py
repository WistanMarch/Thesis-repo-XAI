# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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
                        'IG(Zero)',
                        'EG',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
             }


# Standard Deviation Range
stdev_spread_range = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
# Combine new & old scores or replace all
replace_scores = True


# Apply on SG / SGSQ
use_SGSQ = [
            False,
            True,
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



#--------------------------         CONCATENE ALL SCORES AND SAVE         ---------------------------#


def concatDetailed(method, SGSQ, pathDetailed, pathScores):
    
    print("\t\t Concatenation of all Results")
    
    # pathDetailedScores = pathResults + 'Detailed/'
    SGname = 'SGSQ+' if SGSQ else 'SG+'

    # Empty list for all results
    df_all_results = pd.DataFrame()

    # Get detailed scores filenames
    detailed_scores_files = [f for f in os.listdir(pathDetailed) if (SGname + method + '_' in f)]
    
    # For each file in directory
    for name in detailed_scores_files:
    
        # Load scores dataframe
        df_scores = pd.read_csv(pathDetailed + name, index_col=0).fillna(0)
    
        # Append this DataFrame to the overall DataFrame
        df_all_results = pd.concat([df_all_results, df_scores], ignore_index=True)
        
    # Save overall results file
    df_all_results.to_csv(pathScores + 'Global_Scores_' + SGname + method + '.csv')


#------------------------------------------------------------------------------------------------------#
    
#---------------------------------     FIDELITY SCORES (BLUR INPUT)     -------------------------------#


# Compute Fidelity detailed scores
def Fidelity_blur(net, loader, input_params, SGSQ, paths):
    
    ids = input_params['IDs']
              
    # For each map type
    for map_type in MAP_TYPES:
            
        print("\t\t Map Type", map_type)
    
        # Get files names in correct folder
        pathLoadFolder = paths['SG'] + MAP_TYPES[map_type] + '/' + input_params['Method'] + '/' + input_params['Trial']
        if not SGSQ: filenames = [file for file in os.listdir(pathLoadFolder) if ('SG+' in file)]
        else: filenames = [file for file in os.listdir(pathLoadFolder) if ('SGSQ+' in file)]
            
        # Keep only files for specified noise level values (if values are specified)
        if (len(stdev_spread_range) > 0):
            filenames = [name for lvl in stdev_spread_range for name in filenames if (str(lvl) in name)]
            
        # Extract noise level values / method / SG variant / slice nb from filenames
        split_filenames = [re.split(r"[_+]", n.replace('.nii', '')) for n in filenames]
        df_split = pd.DataFrame(np.array(split_filenames), columns=['seg', '0', 'SG', 'method', 'ID', 'noise'])[['ID', 'noise']]
        df_split['filename'] = filenames
        
        # For each element in loader
        for step, data in enumerate(loader):
            
            # Display id
            print("\t\t\t\t ID", ids[step])
            
            # Load original input / Inference
            in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
            in_numpy = in_tensor.squeeze().cpu().numpy()
            pred_in = nn.functional.softmax(net(in_tensor),dim=1).squeeze().detach().cpu().numpy()
            
            # Load perturbed image
            ni_img_perturbed = nib.load(paths['Perturb'] + 'blur_' + ids[step] + '.nii')
            in_perturbed_tensor = torch.from_numpy(ni_img_perturbed.get_fdata()).to(device).float()
            
            # Get filenames for chosen slice
            df_id = df_split[df_split['ID'] == ids[step]]
            names_id = df_id['filename'].to_list()
                     
            # Detailed scores array
            detail_scores_id = []
        
            # For each file with this slice
            for idx, name in enumerate(names_id):
                
                # Get noise level value
                lvlNb = df_id.iloc[idx]['noise']
                
                # Load attribution map
                ni_map_attr = nib.load(pathLoadFolder + name)
                map_attr_tensor = torch.from_numpy(ni_map_attr.get_fdata()).to(device)
                
                # For each percentage
                for percIdx in range (len(importancePerc)):
                    perc = importancePerc[percIdx]
                    
                    # If 0% then save original prediction
                    if (perc == 0):
                        detail_scores_id.append([pred_in[label], pred_in[label], lvlNb, perc, ids[step], map_type])
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
                        detail_scores_id.append([output_MIF, output_LIF, lvlNb, perc, ids[step], map_type])
            
            # Convert detailed array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores_id, columns=['MIF', 'LIF', 'Noise', 'Percentage', 'ID', 'Map'])
            # os.makedirs(paths['Results'] + 'detailed/', exist_ok=True)
            pathIDScores = paths['Detailed'] + 'blur_' + map_type + name[:name.rfind('_')].replace(MAP_TYPES[map_type], '') + '.csv'
            
            # Look for existing scores file to combine with
            if(os.path.exists(pathIDScores) and not replace_scores):
                DF_old = pd.read_csv(pathIDScores, index_col=0).fillna(0)
                DF_detail_scores = pd.concat([DF_old, DF_detail_scores])

            # Save the dataframe as a csv file
            DF_detail_scores.to_csv(pathIDScores)
                

#----------------------------------------------------------------------------------------------#
    
#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_blur(method, SGSQ, pathScores):
    
    print("\t\t Display Heatmap & Curves")
    
    SGname = 'SGSQ+' if SGSQ else 'SG+'

    # Load global scores DataFrame
    df_all_results = pd.read_csv(pathScores + 'Global_Scores_' + SGname + method + '.csv', index_col=0).fillna(0) 
    
    # Extract lists of noise level & slices values
    noise_lvls = np.unique(df_all_results['Noise'].tolist()).tolist()
    id_values = np.unique(df_all_results['ID'].tolist()).tolist()
    
    # Init heatmap figure
    figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(15, 15))
    figHeatmapStd, axHeatmapStd = plt.subplots(figsize=(15, 15))
    # Init MIF/LIF & Fidelity curves figures
    figMIFLIFCurves, axMIFLIFCurves = plt.subplots(nrows=len(MAP_TYPES), ncols=len(noise_lvls), figsize=(20*len(noise_lvls), 10*len(MAP_TYPES)), squeeze=False)
    figFidCurves, axFidCurves = plt.subplots(nrows=2, ncols=2, figsize=(40, 20), squeeze=False)

    # DataFrame for Fidelity scores
    df_mean = pd.DataFrame(index=noise_lvls, dtype=float)
    df_std = pd.DataFrame(index=noise_lvls, dtype=float)
        
    # For each map type
    for idxMap, map_type in enumerate(MAP_TYPES):
        
        # Extract map type data
        df_one_mtype = df_all_results[df_all_results['Map'] == map_type]
        
        # Init lists
        fidMeanMap = []
        fidStdMap = []
        fidScoresCurves = []
        
        # For each noise level value
        for idxLvl, lvl in enumerate(noise_lvls):
            
            # List of Fidelity scores per slice
            fidScores = []
            # DataFrame of all outputs
            df_preds = pd.DataFrame()
            
            # Extract data for noise level value
            df_one_lvl = df_one_mtype[df_one_mtype['Noise'] == lvl]
        
            # For each slice
            for idxID, ID in enumerate(id_values):
                    
                # Extract data for slice
                df_one_id = df_one_lvl[df_one_lvl['ID'] == ID]
        
                # Extract MIF & LIF results
                MIF_res = list(df_one_id[['MIF']].to_numpy())
                LIF_res = list(df_one_id[['LIF']].to_numpy())
                # Compute area between MIF & LIF curves
                AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
                AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
                Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
                
                # Save output scores (LIF then MIF)
                df_preds_tmp = pd.DataFrame(data=LIF_res+MIF_res, columns=['Output'])
                df_preds_tmp['Perc'] = 2 * importancePerc
                df_preds_tmp['MIF_LIF'] = ['LIF'] * len(importancePerc) + ['MIF'] * len(importancePerc)
                df_preds_tmp['Slice'] = [ID] * len(importancePerc) * 2
                df_preds = pd.concat([df_preds, df_preds_tmp], ignore_index=True)
                fidScores.append(Fid_score)
                fidScoresCurves.append([Fid_score, AUC_LIF, AUC_MIF, ID, lvl])
                
            # Plot MIF & LIF curves
            sns.lineplot(x='Perc', y='Output', hue='MIF_LIF', data=df_preds, ax=axMIFLIFCurves[idxMap][idxLvl]).set(title='MIF & LIF Curves: ' + map_type + ' Noise ' + str(lvl), ylim=(0.0, 1.0))
        
            # Mean & Std of Fidelity scores
            fidMeanMap.append(np.mean(fidScores))
            fidStdMap.append(np.std(fidScores))
                    
        # Save scores for this noise lvl
        df_mean[map_type] = fidMeanMap
        df_std[map_type] = fidStdMap
        df_fid = pd.DataFrame(data=fidScoresCurves, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'ID', 'Noise'])
        
        # Average Fidelity curves plot
        sns.lineplot(x='Noise', y='Fidelity', data=df_fid, ax=axFidCurves[idxMap%2][idxMap//2]).set(title='Fidelity for Map Type: ' + map_type, ylim=(0.0, 1.0))

    # Fill heatmaps
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    sns.heatmap(df_std, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapStd)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
    axHeatmapStd.set_yticklabels(df_std.index.values, rotation=0, rotation_mode='anchor', ha='right')
    # Save heatmap
    figHeatmapMean.savefig(pathScores + 'Fidelity_Heatmap_' + SGname + method + '.tiff')
    figHeatmapStd.savefig(pathScores + 'Fidelity_Heatmap_Std_' + SGname + method + '.tiff')
    # Save curves
    figMIFLIFCurves.tight_layout()
    figMIFLIFCurves.savefig(pathScores + 'Fidelity_MIFLIF_Curves_' + SGname + method + '.tiff')
    figFidCurves.tight_layout()
    figFidCurves.savefig(pathScores + 'Fidelity_Scores_' + SGname + method + '.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():

    # For both SG & SGSQ
    for SGSQ in use_SGSQ:
        
        # For each trial (=fold)
        for trial in trials:
            
            print('Trial ' + str(trial) + ' SGSQ ' + str(SGSQ))
            
            # Trial folder name
            trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
            
            # Create dataset for XAI
            ids, _, loader, net = XAI_dataset(params, trial, device, loading)
            
            # Path for perturbed images
            pathPerturb = loading + 'dataset_crop-x-0-y-0-z-0/data_perturbed/' + trial_folder
            
            # SG & Detailed paths
            pathSG = loading + 'Results/SG_SGSQ_Optim/'
            pathResults = pathSG + 'Metrics/Fidelity_blur/' + trial_folder
            pathDetailed = pathResults + 'detailed/'
            os.makedirs(pathDetailed, exist_ok=True)
            pathScores = pathResults + 'Scores/'
            os.makedirs(pathScores, exist_ok=True)
            
            paths = {
                        'Perturb' : pathPerturb,
                        'SG' : pathSG,
                        'Detailed' : pathDetailed,
                    }
            
            # Range of methods
            for method in (INTERPRET_METHODS):
                print('\t Method', method)
                
                # Input elements dictionary
                input_params = {
                                'IDs' : ids,
                                'Trial' : trial_folder,
                                'Method' : method,
                               }
                
                # For Fidelity with a blurred version of input
                Fidelity_blur(net, loader, input_params, SGSQ, paths)
                
                # Concatenation of all detailed results for the method
                concatDetailed(method, SGSQ, pathDetailed, pathScores)
    
                # Display Fidelity scores
                display_Fidelity_blur(method, SGSQ, pathScores)


# Using the special variable
if __name__=="__main__":
    main()