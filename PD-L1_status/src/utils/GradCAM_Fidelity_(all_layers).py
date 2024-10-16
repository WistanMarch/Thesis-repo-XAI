# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import re
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

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



#-------------------------------------------------------------------------------------------------------------#

#---------------------------------     FIDELITY SCORES (BLUR PERTURBATION)     -------------------------------#


# Compute Fidelity detailed scores
def compute_Fidelity(pathGradCAM):
    
    print("Step 1 : Compute Fidelity")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, net = XAI_dataset(params, trial, device, loading)
        
        # Path for perturbed images
        pathPerturb = loading + 'dataset_crop-x-0-y-0-z-0/data_perturbed/' + trial_folder
        # Path for GradCAM maps
        pathMaps = pathGradCAM + trial_folder + 'Raw_attrs/'
        
        # Detailed files save path
        pathDetailed = pathGradCAM + trial_folder + 'Fidelity/detailed/'
        os.makedirs(pathDetailed, exist_ok=True)
                  
        # Get files names in correct folder
        filenames = [file for file in os.listdir(pathMaps)]
    
        # Split files names
        split_filenames = [re.split(r"[_+]", n.replace('.nii', ''))[3:] for n in filenames]
        df_split = pd.DataFrame(np.array(split_filenames), columns=['ID', 'layer_name'])
        df_split['filename'] = filenames
        
        # For each element in loader
        for step, data in enumerate(loader):
            
            # Display id
            print("\t\t ID", ids[step])
            
            # Load perturbed image
            ni_img_perturbed = nib.load(pathPerturb + 'blur_' + ids[step] + '.nii')
            in_perturbed_tensor = torch.from_numpy(ni_img_perturbed.get_fdata()).to(device).float()
            
            # Get files names for ID
            df_ID = df_split[df_split['ID'] == ids[step]]
            ID_filenames = df_ID['filename'].to_list()
              
            # Detailed scores array
            detail_scores_ID = []
            
            # Load original input / Inference
            in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
            in_numpy = in_tensor.squeeze().cpu().numpy()
            pred_in = nn.functional.softmax(net(in_tensor),dim=1).squeeze().detach().cpu().numpy()
            
            # For each file on this ID
            for idx, name in enumerate(ID_filenames):
                
                # Get layer name
                layerName = df_ID.iloc[idx]['layer_name'].replace('conv', '')
                
                # Load attribution map
                ni_map_attr = nib.load(pathMaps + name)
                map_attr_tensor = torch.from_numpy(ni_map_attr.get_fdata()).to(device)
                
                # For each percentage
                for percIdx in range (len(importancePerc)):
                    perc = importancePerc[percIdx]
                    
                    # If 0% then save original prediction
                    if (perc == 0):
                        detail_scores_ID.append([pred_in[label], pred_in[label], layerName, perc])
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
                        detail_scores_ID.append([output_MIF, output_LIF, layerName, perc])
         
            # Convert detailed array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores_ID, columns=['MIF', 'LIF', 'Layer', 'Percentage'])
            pathIDScores = pathDetailed + 'detailed_ID_' + ids[step] + '.csv'
            
            # Save the dataframe as a csv file
            DF_detail_scores.to_csv(pathIDScores)
                

#--------------------------------------------------------------------------------------------#

#---------------------------------     GLOBAL SCORES     ------------------------------------#


# Concatenate Fidelity scores
def global_scores(pathGradCAM):
   
    print("Step 2 : AUCs & Fidelity scores")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Path for detailed & global scores
        pathFid = pathGradCAM + trial_folder + 'Fidelity/'
        pathDetailed = pathFid + 'detailed/'
        
        # Get files names in correct folder
        filenames = [file for file in os.listdir(pathDetailed) if file.endswith('.csv')]
        
        # Get layers names
        df_tmp = pd.read_csv(pathDetailed + filenames[0], index_col=0).fillna(0)
        layers_names = list(np.unique(df_tmp['Layer'].to_list()))
        
        # Fidelity scores list
        all_scores = []
            
        # For each file in list
        for idx, path in enumerate(filenames):
            # Get scores data
            df_scores = pd.read_csv(pathDetailed + path, index_col=0).fillna(0)
            # Get ID number
            ID = path.replace('detailed_ID_', '').replace('.csv', '')
            
            # For each layer
            for layer in layers_names:
                
                # Extract data for ID
                df_one_layer = df_scores[df_scores['Layer'] == layer][['MIF', 'LIF']]
                
                # Extract MIF & LIF results
                MIF_res = list(df_one_layer[['MIF']].to_numpy())
                LIF_res = list(df_one_layer[['LIF']].to_numpy())
                # Compute area between MIF & LIF curves
                AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
                AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
                Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
                # Add to scores list
                all_scores.append([Fid_score, AUC_LIF, AUC_MIF, layer, ID])
            
        # Save scores as DataFrame
        DF_global_scores = pd.DataFrame(data=all_scores, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'Layer', 'ID'])
        # Save the dataframe as a csv file
        DF_global_scores.to_csv(pathFid + 'Global_Scores.csv')
    
        
#---------------------------------------------------------------------------------------------------------------#
    
#--------------------------------------         FIDELITY DISPLAY         ---------------------------------------#


# Display Fidelity heatmaps & curves
def display_Fidelity(pathGradCAM):
   
    print("Step 3 : Display Heatmap & Curves")
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Path of global scores
        pathFid = pathGradCAM + trial_folder + 'Fidelity/'
        
        # Load all average results
        df_all_results = pd.read_csv(pathFid + 'Global_Scores.csv', index_col=0).fillna(0).sort_values(by=['Layer'])
        
        # Extract list of layers
        layers_names = list(np.unique(df_all_results['Layer'], return_index=False))
        
        # Init heatmap figure
        figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(30, 50))
        # Init average curves figure
        figCurves, axCurves = plt.subplots(nrows=2, ncols=1, figsize=(25, 20), squeeze=False)
        
        # DataFrame for avg values
        df_mean = pd.DataFrame(index=layers_names, columns=['Fidelity'], dtype=float)
        
        # For each layer
        for idx, layer in enumerate(layers_names):
            # Extract layer specific data
            layer_data = df_all_results[df_all_results['Layer'] == layer]['Fidelity'].to_list()
        
            # Compute & save avg
            df_mean.iloc[idx] = [np.mean(layer_data)]
        
        # Average LIF & MIF curves plot
        g1 = sns.lineplot(x='Layer', y='AUC_LIF', data=df_all_results, ax=axCurves[0][0])
        g1.set(title='LIF AUC Curves per Input')
        g1.tick_params(axis='x', rotation=90)
        g2 = sns.lineplot(x='Layer', y='AUC_MIF', data=df_all_results, ax=axCurves[1][0])
        g2.set(title='MIF AUC Curves per Input')
        g2.tick_params(axis='x', rotation=90)
        
        # Fill heatmap
        sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
        axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=45, rotation_mode='anchor', ha='right')
        # Save heatmap
        figHeatmapMean.savefig(pathFid + 'Fidelity_Mean_Heatmap.tiff')
        # Save curves
        figCurves.tight_layout()
        figCurves.savefig(pathFid + 'LIF_MIF_Avg_Curves.tiff')

        
#-------------------------------------------------------------------------------------------------#


  
# Using the special variable
if __name__=="__main__": 

    # Paths
    pathGradCAM = loading + 'Results/GradCAM/'
    
    # Compute Fidelity Scores
    compute_Fidelity(pathGradCAM)
    
    # Call average scores computation
    global_scores(pathGradCAM)

    # Display scores graphs
    display_Fidelity(pathGradCAM)