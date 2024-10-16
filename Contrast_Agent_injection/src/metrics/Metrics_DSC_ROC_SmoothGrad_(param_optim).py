# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""


import numpy as np
import torch
from PIL import Image
import pandas as pd
from itertools import repeat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/SmoothGrad+IG(Black) Parameters Optimization/'

pathTruthMasks = pathRoot + 'dataset_290/Masks/Full/'

# All methods for files loading
PARAMETERS = ['STD Optimization']
# Samples Number Range
SAMPLES_RANGE = [2, 10, 20, 30, 40, 50, 60, 70]
# Standard Deviation Range
STD_RANGE = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

# Percentages of most important regions
importancePerc = [i for i in range(1, 101)]

# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_290.pt')
# Load labels tensor
test_labels = torch.load(pathRoot + 'test_labels_290.pt').numpy()

# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1.4)
    
        
#-----------------------------------------------------#

#------------------     Metrics     ------------------#


# Compute ROC & DSC scores
def compute_ROC_DSC():

    # For each parameter to be optimized
    for param in PARAMETERS:
            
        print("Metrics Parameter", param)
        
        pathDetailedScores = pathResults + param + '/Metrics/'
        
        if (param == 'Samples Optimization'): VALUES = SAMPLES_RANGE
        elif (param == 'STD Optimization'): VALUES = STD_RANGE
                  
        # Global scores array
        global_scores = np.zeros((len(VALUES), len(importancePerc), 3), dtype=float)
        
        # Declare lists that are to be converted into columns
        value_list = [x for item in VALUES for x in repeat(item, len(importancePerc))]
        perc_list = importancePerc * len(VALUES)
    
        # For each slice
        for sliceIdx in range (len(test_images)):
            
            print("\t Slice number", sliceIdx+1)
            
            # Load ground truth mask
            truthMask = np.asarray(Image.open(pathTruthMasks + 'mask_' +str(sliceIdx+1)+ '.tiff').convert('L'))
            truthMask = (truthMask > 127)
                  
            # Detailed scores array
            detail_scores = np.zeros((len(VALUES), len(importancePerc), 3), dtype=float)
                      
            # Range of number of methods
            for valueIdx in range (len(VALUES)):
                
                # Get loading path depending on the method
                pathLoad = pathResults + param + '/Npy Files/Raw_attr_SmoothGrad+IG(Black)_Im_'
                # Load XRAI array
                map_attr = np.load(pathLoad + str(sliceIdx+1) + '_' + str(VALUES[valueIdx]) + '.npy')
                
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
                    
                    # Save in detailed & global arrays
                    detail_scores[valueIdx][percIdx] += [TPR_score, FPR_score, DICE_score]
                    global_scores[valueIdx][percIdx] += [TPR_score, FPR_score, DICE_score]
                    
            # Reshape for save
            detail_scores_2D = detail_scores.reshape(detail_scores.shape[0]*detail_scores.shape[1], -1)
            # Convert array into dataframe
            DF_detail_scores = pd.DataFrame(detail_scores_2D, columns=['TPR','FPR','DICE'])
            # Add lists to DataFrame
            DF_detail_scores['Percentage'] = perc_list
            DF_detail_scores['Value'] = value_list
            # Save the dataframe as a csv file
            DF_detail_scores.to_csv(pathDetailedScores + 'detailed_scores_' + str(sliceIdx+1) + '.csv')
        
        # Average over number of slices
        global_scores /= len(test_images)
            
        # Reshape for save
        global_scores_2D = global_scores.reshape(global_scores.shape[0]*global_scores.shape[1], -1)
        # Convert array into dataframe
        DF_global_scores = pd.DataFrame(global_scores_2D, columns=['TPR','FPR','DICE'])
        # Add lists to DataFrame
        DF_global_scores['Percentage'] = perc_list
        DF_global_scores['Value'] = value_list
        # Save the dataframe as a csv file
        DF_global_scores.to_csv(pathResults + param + ' Global Scores.csv')
    
        
#-------------------------------------------------------------------------------------#
    
#-----------------------------         ROC/AUC         -------------------------------#
    
    
# Display ROC heatmaps & curves
def display_ROC(display_perc=10):
    
    # For each parameter to be optimized
    for param in PARAMETERS:
            
        print("ROC Parameter", param)
        
        # Load scores dataframe
        df_all_results = pd.read_csv(pathResults + param + ' Global Scores.csv', index_col=0).fillna(0)
        # Extract lists of values and percentages
        values = np.unique(df_all_results['Value'].tolist()).tolist()
        
        # Init heatmaps figure
        figAvg, axAvg = plt.subplots(figsize=(8, 15))
        # Init average curves figure
        figROC, axROC = plt.subplots(figsize=(20, 10), squeeze=False)
        
        # DataFrame for AVG
        df_avg = pd.DataFrame(index=values, columns=['Avg'], dtype=float)
        
        # Extract ROC data with <= display_perc% of display (100% = all)
        data_ROC = df_all_results[df_all_results['Percentage'] <= display_perc][['TPR', 'FPR', 'Value']]
    
        # For each parameter value
        for value in values:
            # Extract parameter value data
            data_value = data_ROC[data_ROC['Value'] == value][['TPR', 'FPR']]
        
            # Extract FPR & TPR scores
            fpr_coor = data_value['FPR'].tolist()
            tpr_coor = data_value['TPR'].tolist()
            
            # Sort TPR and FPR lists while keeping reference to each other (because FPR must be in ascending order)
            fpr_coor, tpr_coor = (list(t) for t in zip(*sorted(zip(fpr_coor, tpr_coor))))
            
            # Complete avg DataFrame
            df_avg.at[value, 'Avg'] = metrics.auc(fpr_coor, tpr_coor)
            
        # ROC curves plot
        sns.lineplot(x='FPR', y='TPR', hue='Value', data=data_ROC).set(title='ROC average curves')
                
        # Fill heatmaps
        sns.heatmap(df_avg, annot=True, fmt='.4g', linewidths=.5, ax=axAvg)
        axAvg.set_yticklabels(df_avg.index.values, rotation=0, rotation_mode='anchor', ha='right')
        # # Save heatmaps
        figAvg.savefig(pathResults + param + '/AUC_Avg_Heatmap_' + str(display_perc) + '%.tiff')
        # Save curves
        figROC.tight_layout()
        figROC.savefig(pathResults + param + '/ROC_Avg_curves_' + str(display_perc) + '%.tiff')
    
        
#------------------------------------------------------------------------------------------------------#
    
#--------------------------------------         DSC AVG         ---------------------------------------#
    

# Display DSC heatmaps & curves
def display_DSC(display_perc=100):
    
    # For each parameter to be optimized
    for param in PARAMETERS:
            
        print("DSC Parameter", param)
        
        # Load scores dataframe
        df_all_results = pd.read_csv(pathResults + param + ' Global Scores.csv', index_col=0).fillna(0)
        # Extract lists of values and percentages
        values = np.unique(df_all_results['Value'].tolist()).tolist()
        
        # Init heatmaps figure
        figMaxDSC, axMaxDSC = plt.subplots(figsize=(8, 15))
        # Init average curves figure
        figAvgDSC, axAvgDSC = plt.subplots(figsize=(20, 10), squeeze=False)
        
        # DataFrame for Max
        df_max = pd.DataFrame(index=values, columns=['Max'], dtype=float)
        
        # Extract ROC data with <= display_perc% of display (100% = all)
        data_DSC = df_all_results[df_all_results['Percentage'] <= display_perc][['DICE', 'Value', 'Percentage']]
    
    
        # For each parameter value
        for value in values:
            # Extract method data
            data_value = data_DSC[data_DSC['Value'] == value][['DICE']]
            
            # Complete max DataFrame
            df_max.at[value, 'Max'] = np.max(data_value)
                
        # Average DSC curves plot
        sns.lineplot(x='Percentage', y='DICE', hue='Value', data=data_DSC).set(title='DSC avg curves')
                
        # Fill heatmaps
        sns.heatmap(df_max, annot=True, fmt='.4g', linewidths=.5, ax=axMaxDSC)
        axMaxDSC.set_yticklabels(df_max.index.values, rotation=0, rotation_mode='anchor', ha='right')
        # Save heatmaps
        figMaxDSC.savefig(pathResults + param + '/DSC_Max_Heatmap_' + str(display_perc) + '%.tiff')
        # Save curves
        figAvgDSC.tight_layout()
        figAvgDSC.savefig(pathResults + param + '/DSC_Avg_Curves_' + str(display_perc) + '%.tiff')
    
        
#------------------------------------------------------------------------------------------------------#
    
#------------------------------         ROC/AUC FOR EVERY SLICE         -------------------------------#
    
    
# Display ROC heatmaps & curves
def display_ROC_every_slice(display_perc=10):
    
    # For each parameter to be optimized
    for param in PARAMETERS:
            
        print("ROC every slice Parameter", param)
        
        pathDetailedScores = pathResults + param + '/Metrics/'
        
        # Extract lists of values
        if (param == 'Samples Optimization'): values = SAMPLES_RANGE
        elif (param == 'STD Optimization'): values = STD_RANGE
        values_vote = np.zeros(len(values), dtype=int)
        
        # For each slice
        for sliceIdx in range (len(test_images)):
            
            # Load scores dataframe
            df_all_results = pd.read_csv(pathDetailedScores + 'detailed_scores_' + str(sliceIdx+1) + '.csv', index_col=0).fillna(0)
            
            # DataFrame for AVG
            AUC_avg = np.zeros(len(values), dtype=float)
            
            # Extract ROC data with <= display_perc% of display (100% = all)
            data_ROC = df_all_results[df_all_results['Percentage'] <= display_perc][['TPR', 'FPR', 'Value']]
        
            # For each parameter value
            for value in values:
                # Extract parameter value data
                data_value = data_ROC[data_ROC['Value'] == value][['TPR', 'FPR']]
            
                # Extract FPR & TPR scores
                fpr_coor = data_value['FPR'].tolist()
                tpr_coor = data_value['TPR'].tolist()
                
                # Sort TPR and FPR lists while keeping reference to each other (because FPR must be in ascending order)
                fpr_coor, tpr_coor = (list(t) for t in zip(*sorted(zip(fpr_coor, tpr_coor))))
                AUC_avg[values.index(value)] = metrics.auc(fpr_coor, tpr_coor)
                
            # Find max and argmax
            max_AUC = np.max(AUC_avg)
            max_index = np.argmax(AUC_avg)
            # Get corresponding parameter value in second list
            max_param_value = values[max_index]
            
            # Print results
            print('\t Slice', sliceIdx+1,'Optimal Parameter Value:', max_param_value, 'with AUC =', max_AUC)
            
            # Add one in best scores list
            values_vote[values.index(max_param_value)] += 1
            
        # Transform best scores into DataFrame
        df_best_scores = pd.DataFrame(list(zip(values, values_vote)), columns =['Value', 'Nb Best'])
        # Save DataFrame
        df_best_scores.to_csv(pathResults + param + '/AUC Vote ' + str(display_perc) + '%.csv')
    
        
#------------------------------------------------------------------------------------------------------#
    
#--------------------------------         DSC FOR EVERY SLICE         ---------------------------------#
    

# Display DSC heatmaps & curves
def display_DSC_every_slice(display_perc=100):
    
    # For each parameter to be optimized
    for param in PARAMETERS:
            
        print("DSC every slice Parameter", param)
        
        pathDetailedScores = pathResults + param + '/Metrics/'
        
        # Extract lists of values
        if (param == 'Samples Optimization'): values = SAMPLES_RANGE
        elif (param == 'STD Optimization'): values = STD_RANGE
        values_vote = np.zeros(len(values), dtype=int)
        values_best = np.zeros(len(test_images), dtype=float)
        
        # For each slice
        for sliceIdx in range (len(test_images)):
            
            # Load scores dataframe
            df_all_results = pd.read_csv(pathDetailedScores + 'detailed_scores_' + str(sliceIdx+1) + '.csv', index_col=0).fillna(0)
            
            # Take list of DSC scores and values
            data_DSC = df_all_results['DICE'].tolist()
            data_values = df_all_results['Value'].tolist()
            
            # Find max and argmax
            max_DSC = np.max(data_DSC)
            max_index = np.argmax(data_DSC)
            # Get corresponding parameter value in second list
            max_param_value = data_values[max_index]
            values_best[sliceIdx] = max_param_value
            
            # Print results
            print('\t Slice', sliceIdx+1,'Optimal Parameter Value:', max_param_value, 'with DSC =', max_DSC)
            
            # Add one in best scores list
            values_vote[values.index(max_param_value)] += 1
            
        # Transform best scores into DataFrame
        df_vote_scores = pd.DataFrame(list(zip(values, values_vote)), columns =['Value', 'Nb Best'])
        # Save DataFrame
        df_vote_scores.to_csv(pathResults + param + '/DSC Vote ' + str(display_perc) + '%.csv')
            
        # Transform best scores into DataFrame
        df_best_scores = pd.DataFrame(list(zip(test_labels, values_best)), columns =['Labels', 'Votes'])
        # Save DataFrame
        df_best_scores.to_csv(pathResults + param + '/DSC Detailed Votes ' + str(display_perc) + '%.csv')
        
        
#------------------------------------------------------------------------------------------------------#


  
# Using the special variable
if __name__=="__main__": 
    
    # Compute ROC & DSC Metrics
    compute_ROC_DSC()
    # Display scores graphs
    display_ROC(100)
    display_ROC(10)
    display_DSC()
    
    # Special case: DSC for every slice
    display_ROC_every_slice(100)
    display_ROC_every_slice(10)
    display_DSC_every_slice()