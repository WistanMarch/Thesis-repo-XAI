# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import re
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



#------------------     Parameters     ------------------#


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# All methods for files loading
INTERPRET_METHODS = [
                    'BP',
                    # 'Deconv',
                    'IG(0)',
                    # 'IG(1)',
                    # 'IG(0-1)',
                    # 'IGAW1B0',
                    # 'IGAW0B1',
                    'EG',
                    'GradCAM',
                     ]


# All Map Types to load
MAP_TYPES = {
            'Pix-Abs' : 'Raw_attrs(absolute)',
            'Pix-Orig' : 'Raw_attrs',
            # 'Reg-Abs' : 'XRAI_attrs(absolute)',
            # 'Reg-Orig' : 'XRAI_attrs',
             }


# Standard Deviation Range
stdev_spread_range = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
# Combine new & old scores or replace all
replace_scores = True


# Computing Fidelity for SmoothGrad / SmoothGrad Squared
use_SGSQ = [
                False,
                True,
           ]


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


# Base paths
pathRoot = './'
pathPerturb = pathRoot + 'dataset_260/data_perturbed/'


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#--------------------------         CONCATENE ALL SCORES AND SAVE         ---------------------------#


def concatDetailed(method, SGSQ, pathResults):
    
    print("\t\t Concatenation of all Results")
    
    pathDetailedScores = pathResults + 'Detailed/'
    SGname = 'SGSQ+' if SGSQ else 'SG+'

    # Empty list for all results
    df_all_results = pd.DataFrame()

    # Get detailed scores filenames
    detailed_scores_files = [f for f in os.listdir(pathDetailedScores) if (SGname + method + '_' in f)]
    
    # For each file in directory
    for name in detailed_scores_files:
    
        # Load scores dataframe
        df_scores = pd.read_csv(pathDetailedScores + name, index_col=0).fillna(0)
    
        # Append this DataFrame to the overall DataFrame
        df_all_results = pd.concat([df_all_results, df_scores], ignore_index=True)
        
    # Save overall results file
    df_all_results.to_csv(pathResults + 'Global_Scores_' + SGname + method + '.csv')


#------------------------------------------------------------------------------------------------------#
    
#---------------------------------     FIDELITY SCORES (BLUR INPUT)     -------------------------------#


# Compute Fidelity detailed scores
def Fidelity_blur(model, perturb_images, method, SGSQ, pathNet, pathResults):
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
              
    # For each map type
    for map_type in MAP_TYPES:
            
        print("\t\t Map Type", map_type)
    
        # Get files names in correct folder
        pathLoadFolder = pathNet + MAP_TYPES[map_type] + '/' + method + '/'
        if not SGSQ: filenames = [file for file in os.listdir(pathLoadFolder) if ('SG+' in file)]
        else: filenames = [file for file in os.listdir(pathLoadFolder) if ('SGSQ+' in file)]
            
        # Keep only files for specified noise level values (if values are specified)
        if (len(stdev_spread_range) > 0):
            filenames = [name for lvl in stdev_spread_range for name in filenames if (str(lvl) in name)]
            
        # Extract noise level values / method / SG variant / slice nb from filenames
        split_filenames = [re.split(r"[_+]", n.replace('.npy', '')) for n in filenames]
        df_split = pd.DataFrame(np.array(split_filenames), columns=['seg', '0', 'SG', 'method', '1', 'slice_nb', 'noise'])[['slice_nb', 'noise']]
        df_split['filename'] = filenames
        slices_nb = list(np.unique(df_split['slice_nb'].to_list()))
        
        # For each slice available
        for sliceNb in slices_nb:
            
            if (int(sliceNb)-1 not in excluded_idx):
            
                # Display of the current slice (every 50)
                if (int(sliceNb) == 1 or int(sliceNb) % 50 == 0 or int(sliceNb) == int(slices_nb[-1])): print("\t\t\t Fidelity Slice", int(sliceNb))
                
                # Load perturbed image
                im_perturbed = perturb_images[int(sliceNb)-1].squeeze()
                
                # Get filenames for chosen slice
                df_slice = df_split[df_split['slice_nb'] == sliceNb]
                names_slice = df_slice['filename'].to_list()
                
                # Load input image & corresponding label
                im_tensor = test_images[int(sliceNb)-1].squeeze()
                pred_in = float(torch.sigmoid(model(im_tensor.unsqueeze(0).unsqueeze(0))).detach().cpu().numpy())
                                
                # Detailed scores array
                detail_scores_slice = []
            
                # For each file with this slice
                for idx, name in enumerate(names_slice):
                    
                    # Get noise level value
                    lvlNb = df_slice.iloc[idx]['noise']
                    # Load attribution map
                    map_attr = np.load(pathLoadFolder + name)
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
                            if (np.round(pred_in) == 0): detail_scores_slice.append([1-pred_in, 1-pred_in, lvlNb, perc, int(sliceNb), map_type])
                            else: detail_scores_slice.append([pred_in, pred_in, lvlNb, perc, int(sliceNb), map_type])
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
                            
                            # Compute output and save diff
                            with torch.no_grad():
                                output = torch.sigmoid(model(data_MIF_LIF)).detach().cpu().numpy()
                                
                                if (np.round(pred_in) == 0): output_MIF_LIF.extend(1 - output)
                                else: output_MIF_LIF.extend(output)
                                
                            # Empty batch data
                            data_MIF_LIF = torch.empty((1, 320, 384), device=device)
                                
                    # Save in detailed array
                    detail_scores_slice.extend([[output_MIF_LIF[i*2][0], output_MIF_LIF[i*2+1][0], lvlNb, importancePerc[i+1], int(sliceNb), map_type] for i in range(len(output_MIF_LIF)//2)])
     
                # Convert detailed array into dataframe
                DF_detail_scores = pd.DataFrame(detail_scores_slice, columns=['MIF', 'LIF', 'Noise', 'Percentage', 'Slice', 'Map'])
                os.makedirs(pathResults + 'detailed/', exist_ok=True)
                pathSliceScores = pathResults + 'detailed/blur_' + map_type + '_' + name[:name.rfind('_')] + '.csv'
                
                # Look for existing scores file to combine with
                if(os.path.exists(pathSliceScores) and not replace_scores):
                    DF_old = pd.read_csv(pathSliceScores, index_col=0).fillna(0)
                    DF_detail_scores = pd.concat([DF_old, DF_detail_scores])
    
                # Save the dataframe as a csv file
                DF_detail_scores.to_csv(pathSliceScores)
                

#----------------------------------------------------------------------------------------------#
    
#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_blur(method, SGSQ, pathResults):
    
    print("\t\t Display Heatmap & Curves")
    
    SGname = 'SGSQ+' if SGSQ else 'SG+'

    # Load global scores DataFrame
    df_all_results = pd.read_csv(pathResults + 'Global_Scores_' + SGname + method + '.csv', index_col=0).fillna(0) 
    
    # Extract lists of noise level & slices values
    noise_lvls = np.unique(df_all_results['Noise'].tolist()).tolist()
    slice_values = np.unique(df_all_results['Slice'].tolist()).tolist()
    
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
            for idxSlice, sliceNb in enumerate(slice_values):
                    
                # Extract data for slice
                df_one_slice = df_one_lvl[df_one_lvl['Slice'] == sliceNb]
        
                # Extract MIF & LIF results
                MIF_res = list(df_one_slice[['MIF']].to_numpy())
                LIF_res = list(df_one_slice[['LIF']].to_numpy())
                # Compute area between MIF & LIF curves
                AUC_LIF = metrics.auc(np.asarray(importancePerc)/100, LIF_res)
                AUC_MIF = metrics.auc(np.asarray(importancePerc)/100, MIF_res)
                Fid_score = 1 - (np.abs(1 - AUC_LIF) + np.abs(0.5 - AUC_MIF)) / 1.5
                
                # Save output scores (LIF then MIF)
                df_preds_tmp = pd.DataFrame(data=LIF_res+MIF_res, columns=['Output'])
                df_preds_tmp['Perc'] = 2 * importancePerc
                df_preds_tmp['MIF_LIF'] = ['LIF'] * len(importancePerc) + ['MIF'] * len(importancePerc)
                df_preds_tmp['Slice'] = [sliceNb] * len(importancePerc) * 2
                df_preds = pd.concat([df_preds, df_preds_tmp], ignore_index=True)
                fidScores.append(Fid_score)
                fidScoresCurves.append([Fid_score, AUC_LIF, AUC_MIF, sliceNb, lvl])
                
            # Plot MIF & LIF curves
            sns.lineplot(x='Perc', y='Output', hue='MIF_LIF', data=df_preds, ax=axMIFLIFCurves[idxMap][idxLvl]).set(title='MIF & LIF Curves: ' + map_type + ' Noise ' + str(lvl), ylim=(0.0, 1.0))
        
            # Mean & Std of Fidelity scores
            fidMeanMap.append(np.mean(fidScores))
            fidStdMap.append(np.std(fidScores))
                    
        # Save scores for this noise lvl
        df_mean[map_type] = fidMeanMap
        df_std[map_type] = fidStdMap
        df_fid = pd.DataFrame(data=fidScoresCurves, columns=['Fidelity', 'AUC_LIF', 'AUC_MIF', 'Slice', 'Noise'])
        
        # Average Fidelity curves plot
        sns.lineplot(x='Noise', y='Fidelity', data=df_fid, ax=axFidCurves[idxMap%2][idxMap//2]).set(title='Fidelity for Map Type: ' + map_type, ylim=(0.0, 1.0))

    # Fill heatmaps
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    sns.heatmap(df_std, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapStd)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
    axHeatmapStd.set_yticklabels(df_std.index.values, rotation=0, rotation_mode='anchor', ha='right')
    # Save heatmap
    figHeatmapMean.savefig(pathResults + 'Fidelity_Heatmap_' + SGname + method + '.tiff')
    figHeatmapStd.savefig(pathResults + 'Fidelity_Heatmap_Std_' + SGname + method + '.tiff')
    # Save curves
    figMIFLIFCurves.tight_layout()
    figMIFLIFCurves.savefig(pathResults + 'Fidelity_MIFLIF_Curves_' + SGname + method + '.tiff')
    figFidCurves.tight_layout()
    figFidCurves.savefig(pathResults + 'Fidelity_Scores_' + SGname + method + '.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():

    # For both SG & SGSQ
    for SGSQ in use_SGSQ:

        # For each network
        for arch in networks:
                
            print("Network " + arch + " SGSQ " + str(SGSQ))
            
            # Load model
            model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
            # Load perturbed images
            perturb_images = torch.load(pathPerturb + 'blur_slices_' + arch + '.pt').to(device)
            
            pathNet = pathRoot + 'Results/' + arch + '/SG_SGSQ_Optim/'
            pathResults = pathNet + 'Metrics/Fidelity/'
            os.makedirs(pathResults, exist_ok=True)
            
            # Range of methods
            for method in (INTERPRET_METHODS):
                print('\t Method', method)
                
                # For Fidelity with a blurred version of input
                Fidelity_blur(model, perturb_images, method, SGSQ, pathNet, pathResults)
                
                # Concatenation of all detailed results for the method
                concatDetailed(method, SGSQ, pathResults)
    
                # Display Fidelity scores
                display_Fidelity_blur(method, SGSQ, pathResults)


# Using the special variable
if __name__=="__main__":
    main()