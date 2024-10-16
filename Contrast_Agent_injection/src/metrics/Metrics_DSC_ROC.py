# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from itertools import repeat


#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'
pathTruthMasks = pathRoot + 'dataset_260/Masks/Full/'


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)


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
importancePerc = [i for i in range(1, 101)]



#------------------     DSC / ROC METRICS     ------------------#


# Defining main function
def DSC_ROC(pathResults, pathSaveMetrics):
    
    # For each map type
    for map_type in MAP_TYPES:
            
        print("\t Map Type", map_type)
        
        # Files names depend on value and map types chosen
        detail_filename = "detailed_scores_"
        
        # Path of detailed scores
        pathSaveDetails = pathSaveMetrics + 'Detailed/DSC_ROC/'
        
        # Range of number of methods
        for methodIdx in range (len(INTERPRET_METHODS)):
            print("\t\t", INTERPRET_METHODS[methodIdx])
            
            # Get files names in correct folder
            pathLoadFolder = pathResults + MAP_TYPES[map_type] + '/' + INTERPRET_METHODS[methodIdx] + '/'
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
                
                    # Load ground truth mask
                    truthMask = np.asarray(Image.open(pathTruthMasks + 'mask_' +str(sliceNb)+ '.tiff').convert('L'))
                    truthMask = (truthMask > 127)
                         
                    # Get file name for chosen slice (should be only 1)
                    df_slice = df_split[df_split['slice_nb'] == sliceNb]
                    name_slice = df_slice['filename'].to_list()[0]
                     
                    # Load map
                    map_attr = np.load(pathLoadFolder + name_slice)
                    
                    # Range of chosen percentages
                    for percIdx in range (len(importancePerc)):
                        perc = importancePerc[percIdx]
                        
                        # Show most salient % of the image as mask
                        attr_mask = map_attr >= np.percentile(map_attr, 100-perc)
            
                        # Compute TP / FP / TN / FN
                        nb_TP = np.sum(np.logical_and(truthMask, attr_mask))
                        nb_FP = np.sum(np.logical_and(np.logical_not(truthMask), attr_mask))
                        nb_FN = np.sum(np.logical_and(truthMask, np.logical_not(attr_mask)))
                        nb_TN = np.sum(np.logical_and(np.logical_not(truthMask), np.logical_not(attr_mask)))
                        
                        # ROC values
                        if (nb_TP + nb_FN == 0): TPR_score = 0
                        else: TPR_score = nb_TP / (nb_TP + nb_FN)
                        if (nb_FP + nb_TN == 0): FPR_score = 0
                        else: FPR_score = nb_FP / (nb_FP + nb_TN)
                        
                        # Dice Score
                        if (nb_TP + nb_FP + nb_FN == 0): DICE_score = 0
                        else: DICE_score = (2 * nb_TP) / (2 * nb_TP + nb_FP + nb_FN)
                        
                        # Save in detailed arrays
                        detail_scores.append([TPR_score, FPR_score, DICE_score, perc, sliceNb, map_type, INTERPRET_METHODS[methodIdx]])
                
                else:
                    print("\t\t\t Excluding Image", sliceNb)
                    
            # Convert array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores, columns=['TPR','FPR','DICE', 'Percentage', 'Slice', 'Map', 'Method'])
            # Save the dataframe as a csv file
            os.makedirs(pathSaveDetails, exist_ok=True)
            DF_detail_scores.to_csv(pathSaveDetails + detail_filename + INTERPRET_METHODS[methodIdx] + '_' + map_type + ".csv")


#--------------------------------------------------------------------------------------------------------------#
    

#-----------------------------         GET ALL DETAILED SCORES AND SAVE         -------------------------------#


def concat_detailed(pathSaveMetrics, detailsPath):
    
    print("\t Concatenation of all Results")
    
    pathDetailedScores = pathSaveMetrics + 'Detailed/DSC_ROC/'

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
    df_all_results_sort = df_all_results[df_all_results['Map'].isin(MAP_TYPES)]
    
    # Save overall results file
    df_all_results_sort.to_csv(detailsPath)


#------------------------------------------------------------------------------------------------------#


#---------------------------------         DSC AVG/STD/HIST         -----------------------------------#


def display_DSC(pathSavePlots, detailsPath):
    
    print("\t DSC Plots")
    
    # Load scores dataframe
    df_all_results = pd.read_csv(detailsPath, index_col=0).fillna(0)
    
    # Extract lists of result type, slices and percentages
    results_type = np.unique(df_all_results['Map'].tolist()).tolist()
    slices = np.unique(df_all_results['Slice'].tolist()).tolist()
    percentages = np.unique(df_all_results['Percentage'].tolist()).tolist()
    
    # Init histograms and heatmaps figures
    nbRows = len(INTERPRET_METHODS)
    nbCols = len(results_type)
    fig_dims = (20*nbCols, 10*nbRows)
    figHistDSC, axHistDSC = plt.subplots(nrows=nbRows, ncols=nbCols, figsize=fig_dims, squeeze=False)
    figMaxDSC, axMaxDSC = plt.subplots(figsize=(15, 15))
    figStdDSC, axStdDSC = plt.subplots(figsize=(15, 15))
    figBarplot, axBarplot = plt.subplots(nrows=1, ncols=nbCols, figsize=(12*nbCols, 20), squeeze=False)
    # Init average curves figures
    figAvgDSC, axAvgDSC = plt.subplots(nrows=2, ncols=2, figsize=(40, 20), squeeze=False)
    
    # DataFrames for Max and STD
    df_max = pd.DataFrame(index=INTERPRET_METHODS, columns=results_type, dtype=float)
    df_std = pd.DataFrame(index=INTERPRET_METHODS, columns=results_type, dtype=float)
    
    # For each result type
    for res_type in results_type:
        # Extract result type data
        data_type = df_all_results[df_all_results['Map'] == res_type][['DICE', 'Slice', 'Method']]
        # DSC scores array (one per result type)
        DSC_array_type = np.zeros((len(INTERPRET_METHODS), 100), dtype=float)
        
        array_data_barplot = []
        
        # For each method
        for method in INTERPRET_METHODS:
            # Extract method data
            data_method = data_type[data_type['Method'] == method][['DICE', 'Slice']]
            # DSC scores array (one per method)
            DSC_array_method = []
            
            # For each slice
            for sliceIdx in slices:
                # Extract DSC scores
                data_slice = data_method[data_method['Slice'] == sliceIdx]['DICE'].tolist()
                # Add values to method overall array
                DSC_array_method.append(data_slice)
                
            # Average of DSC scores for each method
            DSC_avg_curve_method = np.mean(DSC_array_method, axis=0)
            DSC_array_type[INTERPRET_METHODS.index(method)] = DSC_avg_curve_method
            
            # Find max of average curve + find related index
            max_value = np.max(DSC_avg_curve_method)
            max_index = np.argmax(DSC_avg_curve_method)
            # Take all slices values at Max percentage and compute std
            max_perc_data = np.array(DSC_array_method)[:, max_index]
            max_perc_std = np.std(max_perc_data)
            
            array_data_barplot.extend(max_perc_data)
            
            # Display Histogram
            idxRow = INTERPRET_METHODS.index(method)
            idxCol = results_type.index(res_type)
            sns.histplot(data=max_perc_data, stat="percent", ax=axHistDSC[idxRow][idxCol]).set(title='Histogram at '+str(max_index+1)+'% (Max DSC value): '+method+' '+res_type, xlim=(0, 1))
            # Complete max and std DataFrames
            df_max.at[method, res_type] = max_value
            df_std.at[method, res_type] = max_perc_std
            
        # Reshape
        DSC_array_2D = DSC_array_type.reshape(DSC_array_type.shape[0]*DSC_array_type.shape[1])
        # Transform to DataFrame
        df_DSC = pd.DataFrame(DSC_array_2D, columns=['DICE'])
        # Add columns
        perc_col = [perc for _ in range(len(INTERPRET_METHODS)) for perc in percentages]
        method_col = [x for item in INTERPRET_METHODS for x in repeat(item, len(percentages))]
        method_label_col = [x for item in INTERPRET_METHODS for x in repeat(item, len(slices))]
        df_DSC['Percentage'] = perc_col
        df_DSC['Method'] = method_col
        df_barplot = pd.DataFrame(array_data_barplot, columns=['DSC'])
        df_barplot['Method'] = method_label_col
        df_barplot['Map'] = [res_type] * len(array_data_barplot)
        # Average DSC curves plot
        idx = results_type.index(res_type)
        sns.lineplot(x='Percentage', y='DICE', hue='Method', data=df_DSC, ax=axAvgDSC[idx%2][idx//2]).set(title='DSC avg curves: '+res_type)
        
        df_max_copy = df_max.sort_values(by=res_type, ascending=False)
        sns.barplot(data=df_barplot, x='DSC', y='Method', hue='Map', ax=axBarplot[0][idx], gap=.1, errorbar='sd', order=df_max_copy.index).set(title='DSC barplot: '+res_type, xlim=(0.0, 0.6))
        # axBarplot[0][idx].tick_params(axis='x', rotation=90)
            
    # Fill heatmaps
    sns.heatmap(df_max, annot=True, fmt='.4g', linewidths=.5, ax=axMaxDSC)
    axMaxDSC.set_yticklabels(df_max.index.values, rotation=0, rotation_mode='anchor', ha='right')
    sns.heatmap(df_std, annot=True, fmt='.4g', linewidths=.5, ax=axStdDSC)
    axStdDSC.set_yticklabels(df_std.index.values, rotation=0, rotation_mode='anchor', ha='right')
    
    # Save histograms and heatmaps
    os.makedirs(pathSavePlots, exist_ok=True)
    figHistDSC.tight_layout()
    figHistDSC.savefig(pathSavePlots + 'DSC_Histograms_at_Max_%.tiff')
    figMaxDSC.savefig(pathSavePlots + 'DSC_Max_Heatmap.tiff')
    figStdDSC.savefig(pathSavePlots + 'DSC_Std_Heatmap.tiff')
    df_max.to_csv(pathSavePlots + 'DSC_Avg.csv')
    df_std.to_csv(pathSavePlots + 'DSC_Std.csv')
    # Save curves
    figAvgDSC.tight_layout()
    figAvgDSC.savefig(pathSavePlots + 'DSC_avg_curves.tiff')
    figBarplot.tight_layout()
    figBarplot.savefig(pathSavePlots + 'DSC_barplots.tiff')

    
#------------------------------------------------------------------------------------------------------#
    

#-----------------------------         ROC/AUC FOR ALL SLICES         -------------------------------#


def display_ROC_AUC(pathSavePlots, detailsPath):
    
    print("\t ROC/AUC Plots")
    
    # Load scores dataframe
    df_all_results = pd.read_csv(detailsPath, index_col=0).fillna(0)
    
    # Extract lists of result type, slices indexes and percentages
    results_type = np.unique(df_all_results['Map'].tolist()).tolist()
    slices = np.unique(df_all_results['Slice'].tolist()).tolist()
    percentages = np.unique(df_all_results['Percentage'].tolist()).tolist()
    
    # Init curves & barplots figures
    nbCols = len(results_type)
    figAvg, axAvg = plt.subplots(figsize=(15, 15))
    figROC, axROC = plt.subplots(nrows=2, ncols=2, figsize=(40, 20), squeeze=False)
    figBarplot, axBarplot = plt.subplots(nrows=1, ncols=nbCols, figsize=(12*nbCols, 20), squeeze=False)
    
    # DataFrames for AVG and STD
    df_avg = pd.DataFrame(index=INTERPRET_METHODS, columns=results_type, dtype=float)
    
    # For each result type
    for res_type in results_type:
        # Extract result type data
        data_type = df_all_results[df_all_results['Map'] == res_type][['TPR', 'FPR', 'Slice', 'Method']]
    
        # FPR / TPR scores array
        FPR_TPR_array = np.zeros((len(INTERPRET_METHODS), 100, 2), dtype=float)
        
        array_data_barplot = []
        
        # For each method
        for method in INTERPRET_METHODS:
            # Extract method data
            data_method = data_type[data_type['Method'] == method][['TPR', 'FPR', 'Slice']]
            # Avg FPR / TPR for method
            FPR_TPR_method = np.zeros((2, 100), dtype=float)
        
            # For each slice
            for sliceIdx in slices:
                # Extract FPR & TPR scores
                data_slice = data_method[data_method['Slice'] == sliceIdx][['TPR', 'FPR']]
                fpr_coor = data_slice['FPR'].tolist()
                tpr_coor = data_slice['TPR'].tolist()
                
                # Add values to average arrays
                FPR_TPR_method += np.array([fpr_coor, tpr_coor]).tolist()
                
            # Average of TPR/FPR scores for each method
            FPR_TPR_method /= len(slices)
            
            # Add values to overall average
            FPR_TPR_array[INTERPRET_METHODS.index(method)] = FPR_TPR_method.T
            
            # Sort FPR and TPR lists while keeping reference to each other (because FPR must be in ascending order)
            fpr, tpr = (list(t) for t in zip(*sorted(zip(FPR_TPR_method[0], FPR_TPR_method[1]))))
            # Compute AUC
            auc = metrics.auc(fpr, tpr)
            
            array_data_barplot.append(auc)
            
            # Complete avg DataFrame
            df_avg.at[method, res_type] = auc
            
        # Reshape
        FPR_TPR_array_2D = FPR_TPR_array.reshape(FPR_TPR_array.shape[0]*FPR_TPR_array.shape[1], -1)
        # Transform to DataFrame
        df_FPR_TPR = pd.DataFrame(FPR_TPR_array_2D, columns=['FPR','TPR'])
        # Add column
        method_col = [x for item in INTERPRET_METHODS for x in repeat(item, len(percentages))]
        df_FPR_TPR['Method'] = method_col
        df_barplot = pd.DataFrame(array_data_barplot, columns=['ROC'])
        df_barplot['Method'] = INTERPRET_METHODS
        df_barplot['Map'] = [res_type] * len(array_data_barplot)
        
        # ROC curves plot
        idx = results_type.index(res_type)
        sns.lineplot(x='FPR', y='TPR', hue='Method', data=df_FPR_TPR, ax=axROC[idx%2][idx//2]).set(title='ROC average curves: '+res_type)
        # AUROC barplot
        df_avg_copy = df_avg.sort_values(by=res_type, ascending=False)
        sns.barplot(data=df_barplot, x='ROC', y='Method', hue='Map', ax=axBarplot[0][idx], gap=.1, order=df_avg_copy.index).set(title='AUROC barplot: '+res_type, xlim=(0.0, 1.0))
        # axBarplot[0][idx].tick_params(axis='x', rotation=90)
        
    # Fill heatmaps
    sns.heatmap(df_avg, annot=True, fmt='.4g', linewidths=.5, ax=axAvg)
    axAvg.set_yticklabels(df_avg.index.values, rotation=0, rotation_mode='anchor', ha='right')
    
    # Save histograms and heatmaps
    os.makedirs(pathSavePlots, exist_ok=True)
    figAvg.savefig(pathSavePlots + 'ROC_Avg_Heatmap.tiff')
    df_avg.to_csv(pathSavePlots + 'ROC_Avg.csv')
    # Save curves
    figROC.tight_layout()
    figROC.savefig(pathSavePlots + 'ROC_Avg_curves.tiff')
    figBarplot.tight_layout()
    figBarplot.savefig(pathSavePlots + 'ROC_AUC_barplots.tiff')

    
#------------------------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/'
        pathSaveMetrics = pathResults + 'Metrics/'
        pathSavePlots = pathSaveMetrics + 'Graphs/DSC_ROC/'
        # Path for final detailed scores file
        detailsPath = pathSaveMetrics + 'all_detailed_scores.csv'
        
        # # Compute DSC & ROC scores
        # DSC_ROC(pathResults, pathSaveMetrics)
        
        # # Concatenation of all detailed scores files
        # concat_detailed(pathSaveMetrics, detailsPath)
        
        # Metrics computation and display
        display_DSC(pathSavePlots, detailsPath)
        display_ROC_AUC(pathSavePlots, detailsPath)

            
# Using the special variable
if __name__=="__main__": 
    main()