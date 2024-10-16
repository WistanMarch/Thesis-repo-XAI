# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import re
import numpy as np
import torch
import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns



#------------------     Parameters     ------------------#


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                262,
                ]


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Base paths
pathRoot = './'
pathTruthMasks = pathRoot + 'dataset_290/Masks/Full/'
pathPerturb = pathRoot + 'dataset_290/data_perturbed/'


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)
    
        
#-------------------------------------------------------------------------------------------------------------#

#---------------------------------     FIDELITY SCORES (BLUR PERTURBATION)     -------------------------------#


# Compute Fidelity detailed scores
def compute_Fidelity(pathRaw, pathResults):
    
    print("\t Step 1 : Compute Fidelity")
    
    # Load model
    model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_290.pt').to(device)
    # Load perturbed images
    perturb_images = torch.load(pathPerturb + 'blur_slices_' + arch + '.pt').to(device)
    
    # Detailed files save path
    pathDetailed = pathResults + 'detailed/'
    os.makedirs(pathDetailed, exist_ok=True)
                  
    # Get files names in correct folder
    filenames = [file for file in os.listdir(pathRaw)]
    
    # Extract slices & layers nb from filenames
    split_filenames = [re.split(r"[_+]", n.replace('.npy', ''))[3:] for n in filenames]
    df_split = pd.DataFrame(np.array(split_filenames), columns=['slice_nb', 'layer_name'])
    df_split['filename'] = filenames
    slices_nb = list(np.unique(df_split['slice_nb'].to_list()))
    # Remove excluded slice indices
    for idx in excluded_idx:
        slices_nb.remove(str(idx))
    
    # For each slice available
    for sliceNb in slices_nb:
        
        # Display of the current slice (every 50)
        if (int(sliceNb)+1 == 1 or (int(sliceNb)+1) % 50 == 0 or int(sliceNb)+1 == int(slices_nb[-1])): print("\t\t Running Slice", int(sliceNb)+1)
        
        # Load perturbed images
        im_perturbed = perturb_images[int(sliceNb)].squeeze()
        
        # Get files names for chosen slice
        df_slice = df_split[df_split['slice_nb'] == sliceNb]
        names_slice = df_slice['filename'].to_list()
        
        # Detailed scores array
        detail_scores_slice = []
        
        # Load input image & infer
        im_tensor = test_images[int(sliceNb)].squeeze()
        pred_in = float(torch.sigmoid(model(im_tensor.unsqueeze(0).unsqueeze(0))).detach().cpu().numpy())
        
        # For each file on this slice
        for idx, name in enumerate(names_slice):
            
            # Get layer number
            layerName = df_slice.iloc[idx]['layer_name']
            
            # Load attribution map
            map_attr = np.load(pathRaw + name)
            map_attr_tensor = torch.from_numpy(map_attr).to(device)
                  
            # Empty tensor on GPU
            data_MIF_LIF = torch.empty((1, 320, 384), device=device)
            
            # Batch size definition
            batch_size = 50
            
            # Predictions list
            output_MIF_LIF = []
            
            # For each percentage
            for percIdx in range (len(importancePerc)):
                perc = importancePerc[percIdx]
                
                # If 0% then save original prediction
                if (perc == 0):
                    if (np.round(pred_in) == 0): detail_scores_slice.append([1-pred_in, 1-pred_in, layerName, perc])
                    else: detail_scores_slice.append([pred_in, pred_in, layerName, perc])
                else:
                    
                    ################      Compute Most Important First      ################
            
                    masked_input = torch.clone(im_tensor)
                    # Hide most salient % of the image
                    mask = map_attr_tensor >= torch.quantile(map_attr_tensor, (100-perc)/100)
                    masked_input[mask] = im_perturbed[mask]
        
                    # Add masked_input to dataset
                    data_MIF_LIF = torch.cat((data_MIF_LIF, masked_input.unsqueeze(0)), 0)
                    
                    ################      Compute Least Important First      ################
            
                    masked_input = torch.clone(im_tensor)
                    # Hide least salient % of the image
                    mask = map_attr_tensor <= torch.quantile(map_attr_tensor, perc/100)
                    masked_input[mask] = im_perturbed[mask]
        
                    # Add masked_input to dataset
                    data_MIF_LIF = torch.cat((data_MIF_LIF, masked_input.unsqueeze(0)), 0)
                    
                    #########################################################################
                
                # If we have a full batch or it is the end of the percentages
                if (len(data_MIF_LIF) == batch_size+1 or perc == importancePerc[-1]):
                    
                    data_MIF_LIF = data_MIF_LIF[1:].unsqueeze(1)
                    
                    # Compute output and save outputs
                    with torch.no_grad():
                        output = torch.sigmoid(model(data_MIF_LIF)).detach().cpu().numpy()
                        
                        if (np.round(pred_in) == 0): output_MIF_LIF.extend(1 - output)
                        else: output_MIF_LIF.extend(output)
                    
                    # Empty batch data
                    data_MIF_LIF = torch.empty((1, 320, 384), device=device)
                        
            # Save in detailed array
            detail_scores_slice.extend([[output_MIF_LIF[i*2][0], output_MIF_LIF[i*2+1][0], layerName, importancePerc[i+1]] for i in range(len(output_MIF_LIF)//2)])
 
        # Convert detailed array into dataframe
        DF_detail_scores = pd.DataFrame(detail_scores_slice, columns=['MIF', 'LIF', 'Layer', 'Percentage'])
        pathSliceScores = pathDetailed + 'detailed_Slice_' + sliceNb + '.csv'
        
        # Save the dataframe as a csv file
        DF_detail_scores.to_csv(pathSliceScores)
                

#--------------------------------------------------------------------------------------------#

#---------------------------------     GLOBAL SCORES     ------------------------------------#


# Concatenate Fidelity scores
def global_scores(pathResults):
   
    print("\t Step 2 : Average scores")
    
    # Path for detailed scores
    pathDetailed = pathResults + 'detailed/'
    
    # Get files names in correct folder
    filenames = [file for file in os.listdir(pathDetailed) if file.endswith('.csv')]
    
    # Get number of layers and slices
    df_tmp = pd.read_csv(pathDetailed + filenames[0], index_col=0).fillna(0)
    layers_names = list(np.unique(df_tmp['Layer'].to_list()))
    slices_nb = [name[name.rfind("_")+1:name.rfind(".")] for name in filenames]
        
    # Fidelity scores list
    all_scores = []
        
    # For each file in list
    for idx, path in enumerate(filenames):
        # Get scores data
        df_scores = pd.read_csv(pathDetailed + path, index_col=0).fillna(0)
        # Get slice number
        sliceNb = slices_nb[idx]
            
        # For each layer number
        for layer in layers_names:
            
            # Extract data for slice
            df_one_layer = df_scores[df_scores['Layer'] == layer][['LIF', 'MIF']]
            layer_idx, layer_name = re.split(r"[-]", layer)
            
            # Extract MIF & LIF results
            LIF_res = list(df_one_layer[['LIF']].to_numpy())
            MIF_res = list(df_one_layer[['MIF']].to_numpy())
            # Compute area between MIF & LIF curves
            AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
            AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
            Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
            # Add to scores list
            all_scores.append([Fid_score, AUC_LIF, AUC_MIF, layer_idx, layer_name, sliceNb])
        
    # Save scores as DataFrame
    DF_global_scores = pd.DataFrame(data=all_scores, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'Layer_Idx', 'Layer_Name', 'Slice'])
    # Save the dataframe as a csv file
    DF_global_scores.to_csv(pathResults + 'Global_Scores.csv')
    
        
#---------------------------------------------------------------------------------------------------------------#
    
#--------------------------------------         FIDELITY DISPLAY         ---------------------------------------#


# Display Fidelity heatmaps & curves
def display_Fidelity(pathResults):
    
    print("\t Step 3 : Display Heatmap & Curves")
    
    # Load all average results
    df_all_results = pd.read_csv(pathResults + 'Global_Scores.csv', index_col=0).fillna(0).sort_values(by=['Layer_Idx'])
    
    # Extract list of layers
    _, index = np.unique(df_all_results['Layer_Idx'], return_index=True)
    layers_names = [df_all_results.iloc[idx]['Layer_Name'] for idx in index]
    
    # Init heatmap figure
    figHeatmapMean, axHeatmapMean = plt.subplots(figsize=(30, 50))
    # Init average curves figure
    figCurves, axCurves = plt.subplots(nrows=2, ncols=1, figsize=(20, 20), squeeze=False)
    
    # DataFrame for avg values
    df_mean = pd.DataFrame(index=layers_names, columns=['Fidelity'], dtype=float)
    
    # For each layer
    for idx, layer in enumerate(layers_names):
        # Extract layer specific data
        layer_data = df_all_results[df_all_results['Layer_Name'] == layer]['Fidelity'].to_list()
        
        # Compute & save avg
        df_mean.iloc[idx] = [np.mean(layer_data)]
    
    # Average LIF & MIF curves plot
    sns.lineplot(x='Layer_Name', y='AUC_LIF', data=df_all_results, ax=axCurves[0][0]).set(title='LIF AUC Curves per Slice')
    axCurves[0][0].tick_params(axis='x', rotation=90)
    sns.lineplot(x='Layer_Name', y='AUC_MIF', data=df_all_results, ax=axCurves[1][0]).set(title='MIF AUC Curves per Slice')
    axCurves[1][0].tick_params(axis='x', rotation=90)
            
    # Fill heatmap
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=45, rotation_mode='anchor', ha='right')
    # Save heatmap
    figHeatmapMean.savefig(pathResults + 'Fidelity_Mean_Heatmap.tiff')
    # Save curves
    figCurves.tight_layout()
    figCurves.savefig(pathResults + 'LIF_MIF_Avg_Curves.tiff')
    
        
#-------------------------------------------------------------------------------------------------#


  
# Using the special variable
if __name__=="__main__": 

    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathNet = pathRoot + 'Results/GradCAM/' + arch + '/'
        pathRaw = pathNet + 'Raw_attrs/'
        pathResults = pathNet + 'Fidelity/'
        os.makedirs(pathResults, exist_ok=True)
        
        # Compute Fidelity Scores
        compute_Fidelity(pathRaw, pathResults)
        
        # Call average scores computation
        global_scores(pathResults)
    
        # Display scores graphs
        display_Fidelity(pathResults)