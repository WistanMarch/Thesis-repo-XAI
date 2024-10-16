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


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Fixed values to use for perturbation
PERTURBATION_VALUES = [
                        0.0,
                        0.5,
                        1.0,
                        "InputAvg",
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
def Fidelity_fixed(pathNet, model, test_images, test_labels, perturb_value):
    
    # Paths
    pathResults = pathNet + 'Metrics/Detailed/Fidelity/' + str(perturb_value) + '/'
    os.makedirs(pathResults, exist_ok=True)
    
    # Range of methods
    for method in (INTERPRET_METHODS):
        print('\t\t Method', method)
                  
        # For each map type
        for map_type in MAP_TYPES:
                
            print("\t\t\t Map Type", map_type)
        
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
                
                    # # Display of the current slice (every 50)
                    # if (int(sliceNb) == 1 or int(sliceNb) % 50 == 0 or int(sliceNb) == int(slices_nb[-1])): print("\t\t\t Running Slice", int(sliceNb))
                    
                    # Load corresponding label
                    label = test_labels[int(sliceNb)-1]
                    
                    # Get file name for chosen slice (should be only 1)
                    df_slice = df_split[df_split['slice_nb'] == sliceNb]
                    name_slice = df_slice['filename'].to_list()[0]
                    
                    # Load input image & infer
                    im_tensor = test_images[int(sliceNb)-1].squeeze()
                    pred_in = float(torch.sigmoid(model(im_tensor.unsqueeze(0).unsqueeze(0))).detach().cpu().numpy())
                    # Correct or not
                    acc_in = np.round(pred_in) == label
                    
                    # Generate fixed value perturbation image (depending on perturb_value)
                    if (perturb_value == 'InputAvg'):
                        avg_value = np.mean(im_tensor.detach().cpu().numpy())
                        im_perturbed = torch.full(im_tensor.shape, avg_value).to(device)
                    else:
                        im_perturbed = torch.full(im_tensor.shape, perturb_value).to(device)
                    
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
                        
                        # If 0% then save if correct prediction
                        if (perc == 0):
                            detail_scores.append([acc_in, acc_in, perc, int(sliceNb), map_type, method])
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
                            
                            # Compute output and save if correct prediction
                            with torch.no_grad():
                                output = torch.sigmoid(model(data_MIF_LIF)).detach().cpu().numpy()
                                
                                y_correct = np.round(output) == label
                                output_MIF_LIF.extend(y_correct)
                            
                            # Empty batch data
                            data_MIF_LIF = torch.empty((1, 320, 384), device=device)
                                
                    # Save in detailed array
                    detail_scores.extend([[output_MIF_LIF[i*2][0], output_MIF_LIF[i*2+1][0], importancePerc[i+1], int(sliceNb), map_type, method] for i in range(len(output_MIF_LIF)//2)])
     
            # Convert detailed array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores, columns=['MIF', 'LIF', 'Percentage', 'Slice', 'Map', 'Method'])
            pathScores = pathResults + 'detailed_' + str(perturb_value) + '_' + method + '_' + map_type + '.csv'
            
            # Save the dataframe as a csv file
            DF_detail_scores.to_csv(pathScores)
                

#----------------------------------------------------------------------------------------------------#

#--------------------------         CONCATENATE ALL SCORES AND SAVE         ---------------------------#
        

def concatDetailed(pathSaveMetrics, detailsPath, perturb_value):
    
    print("\t\t Concatenation of all Results")
    
    pathDetailedScores = pathSaveMetrics + 'Detailed/Fidelity/' + str(perturb_value) + '/'

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
def display_Fidelity_fixed(pathSavePlots, detailsPath, perturb_value):
    
    print("\t\t Display Heatmap & Curves")
    
    # Load global scores DataFrame
    df_all_results = pd.read_csv(detailsPath, index_col=0).fillna(0)
    
    # Extract list of slices indexes
    slice_values = np.unique(df_all_results['Slice'].tolist()).tolist()
    
    # # Create short version of methods names
    # methods_labels = [''.join(c for c in method if c.isupper() or c.isdigit()) for method in INTERPRET_METHODS]
    
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
            MIF_array_method = df_one_method[['MIF']].to_numpy().reshape(len(slice_values), len(importancePerc))
            MIF_accuracy = np.sum(MIF_array_method, axis=0) / len(slice_values)
            
            # Extract LIF method data
            LIF_array_method = df_one_method[['LIF']].to_numpy().reshape(len(slice_values), len(importancePerc))
            LIF_accuracy = np.sum(LIF_array_method, axis=0) / len(slice_values)
            
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
        x_lim = (-0.3, 0.501)
        df_mean_copy = df_mean.sort_values(by=map_type, ascending=False)
        sns.barplot(data=df_fid, x='Fidelity', y='Method', hue='Map', ax=axBarplot[0][idxMap], gap=.1, order=df_mean_copy.index).set(title='Fidelity barplot: ' + map_type, xlim=x_lim)
        axBarplot[0][idxMap].set_xticks(np.arange(x_lim[0], x_lim[1], 0.1))

    # Fill heatmaps
    sns.heatmap(df_mean, annot=True, fmt='.4g', linewidths=.5, ax=axHeatmapMean)
    axHeatmapMean.set_yticklabels(df_mean.index.values, rotation=0, rotation_mode='anchor', ha='right')
    
    # Save heatmaps
    figHeatmapMean.savefig(pathSavePlots + 'Fidelity_' + str(perturb_value) + '_Avg_Heatmap.tiff')
    df_mean.to_csv(pathSavePlots + 'Fidelity_' + str(perturb_value) + '_Avg.csv')
    # Save curves
    figMIFLIFCurves.tight_layout()
    figMIFLIFCurves.savefig(pathSavePlots + 'Fidelity_' + str(perturb_value) + '_MIFLIF_Curves.tiff')
    figBarplot.tight_layout()
    figBarplot.savefig(pathSavePlots + 'Fidelity_' + str(perturb_value) + '_Barplots.tiff')
    
    
#-------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # For each perturbation fixed value
        for perturb_value in PERTURBATION_VALUES:
            
            print("\t Perturbation Value " + str(perturb_value))
            
            pathResults = pathRoot + 'Results/' + arch + '/'
            pathSaveMetrics = pathResults + 'Metrics/'
            pathSavePlots = pathSaveMetrics + 'Graphs/Fidelity_' + str(perturb_value) + '/'
            os.makedirs(pathSavePlots, exist_ok=True)
            # Path for final detailed scores file
            detailsPath = pathSaveMetrics + 'all_fidelity_' + str(perturb_value) + '_scores.csv'
            
            # Load model
            model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
            # Images chosen for application of saliency maps
            test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
            # Load labels of input images
            test_labels = torch.load(pathRoot + 'test_labels_260.pt').numpy()
            
            # # For Fidelity with a blurred version of input
            # Fidelity_fixed(pathResults, model, test_images, test_labels, perturb_value)
            
            # Concatenation of all detailed scores files
            concatDetailed(pathSaveMetrics, detailsPath, perturb_value)
            
            # Fidelity Metric computation and display
            display_Fidelity_fixed(pathSavePlots, detailsPath, perturb_value)

            
# Using the special variable
if __name__=="__main__": 
    main()