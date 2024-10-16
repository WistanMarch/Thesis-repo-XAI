# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'
pathPerturb = pathRoot + 'dataset_260/data_perturbed/'


# List of trained networks
networks = [
            "resnet",
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
                        'GradCAM',
                        'EG',
                        'Random',
                        # 'ImgReg+BP',
                        # 'ImgReg+IG(0)',
                        # 'ImgReg+EG',
                        # 'ImgReg+GradCAM',
                        # 'SG+BP',
                        # 'SG+IG(0)',
                        # 'SG+GradCAM',
                        # 'SG+EG',
                        # 'SGSQ+BP',
                        # 'SGSQ+IG(0)',
                        # 'SGSQ+GradCAM',
                        # 'SGSQ+EG',
                        # 'Avg(BP.IG(0).GradCAM.EG)',
                        # 'Product(BP.IG(0).GradCAM.EG)',
                        # 'Avg(IG(0).GradCAM)',
                        # 'Product(IG(0).GradCAM)',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
             }


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)



#---------------------------------     FIDELITY SCORES (BLUR PERTURBATION)     -------------------------------#


# Compute Fidelity detailed scores
def Fidelity_blur(pathNet, model, test_images, perturb_images):
    
    # Paths
    pathResults = pathNet + 'Metrics/Detailed/Fidelity/Blur/'
    os.makedirs(pathResults, exist_ok=True)
    
    # Range of methods
    for method in (INTERPRET_METHODS):
        print('\t Method', method)
                  
        # For each map type
        for map_type in MAP_TYPES:
                
            print("\t\t Map Type", map_type)
        
            # Get files names in correct folder
            pathLoadFolder = pathNet + MAP_TYPES[map_type] + '/' + method + '/'
            filenames = [file for file in os.listdir(pathLoadFolder)]
            
            # Extract slice nb from filenames
            split_filenames = [n[n.rfind('_')+1 : n.rfind('.')] for n in filenames]
            df_split = pd.DataFrame(np.array(split_filenames), columns=['slice_nb'])
            df_split['filename'] = filenames
            slices_nb = list(np.unique(df_split['slice_nb'].to_list()))
            
            # Detailed scores array
            detail_scores = []
            
            # For each slice available
            for sliceNb in slices_nb:
                
                # Compute if not excluded slice
                if (int(sliceNb)-1 not in excluded_idx):
                    
                    # Load perturbed image
                    im_perturbed = perturb_images[int(sliceNb)-1].squeeze()
                    
                    # Get file name for chosen slice (should be only 1)
                    df_slice = df_split[df_split['slice_nb'] == sliceNb]
                    name_slice = df_slice['filename'].to_list()[0]
                    
                    # Load input image & infer
                    im_tensor = test_images[int(sliceNb)-1].squeeze()
                    pred_in = float(torch.sigmoid(model(im_tensor.unsqueeze(0).unsqueeze(0))).detach().cpu().numpy())
                    
                    # Load attribution map
                    map_attr = np.load(pathLoadFolder + name_slice)
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
                            if (np.round(pred_in) == 0): detail_scores.append([1-pred_in, 1-pred_in, perc, int(sliceNb), map_type, method])
                            else: detail_scores.append([pred_in, pred_in, perc, int(sliceNb), map_type, method])
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
                    detail_scores.extend([[output_MIF_LIF[i*2][0], output_MIF_LIF[i*2+1][0], importancePerc[i+1], int(sliceNb), map_type, method] for i in range(len(output_MIF_LIF)//2)])
     
            # Convert detailed array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores, columns=['MIF', 'LIF', 'Percentage', 'Slice', 'Map', 'Method'])
            pathScores = pathResults + 'detailed_blur_' + method + '_' + map_type + '.csv'
            
            # Save the dataframe as a csv file
            DF_detail_scores.to_csv(pathScores)
                

#----------------------------------------------------------------------------------------------------#

#--------------------------         CONCATENATE ALL SCORES AND SAVE         ---------------------------#


def concatDetailed(pathSaveMetrics, detailsPath):
    
    print("\t Concatenation of all Results")
    
    pathDetailedScores = pathSaveMetrics + 'Detailed/Fidelity/Blur/'

    # Empty list for all results
    df_all_results = pd.DataFrame()

    # Get detailed scores filenames
    detailed_scores_files = [f for f in os.listdir(pathDetailedScores) if os.path.isfile(os.path.join(pathDetailedScores, f))]
    
    # For each file in directory
    for name in detailed_scores_files:
    
        # Load scores dataframe
        df_scores = pd.read_csv(pathDetailedScores + name, index_col=0).fillna(0)
    
        # Append this DataFrame to the overall DataFrame
        df_all_results = pd.concat([df_all_results, df_scores], ignore_index=True)
    
    df_all_results_sort = df_all_results[df_all_results['Method'].isin(INTERPRET_METHODS)]
    
    # Save overall results file
    df_all_results_sort.to_csv(detailsPath)


#----------------------------------------------------------------------------------------------#

#--------------------------         FIDELITY DISPLAY (BLUR)         ---------------------------#


# Display Fidelity Heatmap & Curves
def display_Fidelity_blur(pathSavePlots, detailsPath):
    
    print("\t Display Figures")
    
    # Load global scores DataFrame
    df_all_results = pd.read_csv(detailsPath, index_col=0).fillna(0)
    
    # Extract list of slices indexes
    slice_values = np.unique(df_all_results['Slice'].tolist()).tolist()
    
    # # Create short version of methods names
    # methods_labels = [''.join(c for c in method if c.isupper() or c.isdigit()) for method in INTERPRET_METHODS]
    
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
            
            # For each slice
            for idxSlice, sliceNb in enumerate(slice_values):
                    
                # Extract data for slice
                df_one_slice = df_one_method[df_one_method['Slice'] == sliceNb]
        
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
                fidScoresCurves.append([Fid_score, AUC_LIF, AUC_MIF, sliceNb, method])
                
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
        df_fid.to_csv(pathSavePlots + 'Fidelity_per_slice_' + map_type + '.csv')
        
        # Plot barplots
        df_mean_copy = df_mean.sort_values(by=map_type, ascending=False)
        sns.barplot(data=df_fid, x='Fidelity', y='Method', hue='Map', ax=axBarplot[0][idxMap], gap=.1, errorbar='sd', order=df_mean_copy.index).set(xlim=(0.3, 1.0))
        # Add title if needed: "title='Fidelity barplot: ' + map_type, "

    # Create overall barplot
    df_mean_order = df_mean.sort_values(by=next(iter(MAP_TYPES)), ascending=False)
    sns.barplot(data=df_bars, x='Fidelity', y='Method', hue='Map', gap=.1, ax=axBarAll[0,0], errorbar='sd', order=df_mean_order.index).set(xlim=(0.2, 1.0))
    # Fill heatmaps
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    sns.heatmap(df_std, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapStd)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
    axHeatmapStd.set_yticklabels(df_std.index.values, rotation=0, rotation_mode='anchor', ha='right')
    
    # Save overall barplot
    figBarAll.tight_layout()
    figBarAll.savefig(pathSavePlots + 'Fidelity_Barplot_combined.tiff')
    # Save heatmaps
    figHeatmapMean.savefig(pathSavePlots + 'Fidelity_Blur_Avg_Heatmap.tiff')
    figHeatmapStd.savefig(pathSavePlots + 'Fidelity_Blur_Std_Heatmap.tiff')
    df_mean.to_csv(pathSavePlots + 'Fidelity_Blur_Avg.csv')
    df_std.to_csv(pathSavePlots + 'Fidelity_Blur_Std.csv')
    # Save curves
    figMIFLIFCurves.tight_layout()
    figMIFLIFCurves.savefig(pathSavePlots + 'Fidelity_Blur_MIFLIF_Curves.tiff')
    figBarplot.tight_layout()
    figBarplot.savefig(pathSavePlots + 'Fidelity_Blur_Barplots.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
        
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/'
        pathSaveMetrics = pathResults + 'Metrics/'
        pathSavePlots = pathSaveMetrics + 'Graphs/Fidelity_Blur/'
        os.makedirs(pathSavePlots, exist_ok=True)
        # Path for final detailed scores file
        detailsPath = pathSaveMetrics + 'all_fidelity_blur_scores.csv'
        
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        # Images chosen for application of saliency maps
        test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
        # Load perturbed images
        perturb_images = torch.load(pathPerturb + 'blur_slices_' + arch + '.pt').to(device)
        
        # # For Fidelity with a blurred version of input
        # Fidelity_blur(pathResults, model, test_images, perturb_images)
        
        # # Concatenation of all detailed scores files
        # concatDetailed(pathSaveMetrics, detailsPath)
        
        # Fidelity Metric computation and display
        display_Fidelity_blur(pathSavePlots, detailsPath)

            
# Using the special variable
if __name__=="__main__": 
    main()