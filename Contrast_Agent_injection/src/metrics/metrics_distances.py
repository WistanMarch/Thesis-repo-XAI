# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:32:59 2023

@author: Wistan
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cityblock



#------------------     Parameters     ------------------#



# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            'pix_orig' : 'Raw_attrs',
            'reg_abs' : 'XRAI_attrs(absolute)',
            'reg_orig' : 'XRAI_attrs',
             }


# Base paths
pathRoot = './'


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=1)



#--------------------------         RESHAPE DATAFRAME         ---------------------------#
        

def reshape_DF(df, name):
        
    # Convert to array & flatten
    array_flat = np.array(df).flatten()
    
    # Concatenate methods & map types names
    fullnames = [method+'_'+mapType for method in df.index for mapType in df.columns]
    
    # Flattened DataFrame
    df_reshaped = pd.DataFrame(data=array_flat, index=fullnames, columns=[name])
    
    # Return
    return df_reshaped
        


#----------------------------------------------------------------------------------------#


#----------------------------         PARETO FRONT         ------------------------------#
        

def pareto(df, method1, method2):
    
    # Extract X & Y data
    Xs = df[method1].to_list()
    Ys = df[method2].to_list()
    
    # Sort list according to Xs
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=True)
    
    # First point in sorted list
    pareto_points = [sorted_list[0]]
    pareto_front = df[df[method1] == pareto_points[0][0]].index.to_list()
    
    # For each point
    for pair in sorted_list[1:]:
        # Append if Y coordinate is better than previous in front
        if pair[1] >= pareto_points[-1][1]:
            pareto_points.append(pair)
            pareto_front.extend(df[df[method1] == pareto_points[-1][0]].index.to_list())
    
    # Return
    return pareto_front
        


#----------------------------------------------------------------------------------------#


#-------------------------         EUCLIDEAN DISTANCE         ---------------------------#
        

def euclid_dist(df, method1, method2, best_point):
    
    # Extract X & Y data
    Xs = df[method1].to_list()
    Ys = df[method2].to_list()
    
    # Init list
    distances = []
    
    # For each point
    for i in range(len(Xs)):
        
        # Create point
        point = np.array((Xs[i], Ys[i]))
    
        # Compute Euclidean distance & append
        dist = np.linalg.norm(best_point - point)
        distances.append(dist)
    
    # Return
    return distances
        


#----------------------------------------------------------------------------------------#


#-------------------------         MANHATTAN DISTANCE         ---------------------------#
        

def manhat_dist(df, method1, method2, best_point):
    
    # Extract X & Y data
    Xs = df[method1].to_list()
    Ys = df[method2].to_list()
    
    # Init list
    distances = []
    
    # For each point
    for i in range(len(Xs)):
        
        # Create point
        point = np.array((Xs[i], Ys[i]))
        
        # Compute Manhattan distance & append
        dist = cityblock(best_point, point)
        distances.append(dist)
    
    # Return
    return distances



#----------------------------------------------------------------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # File paths
        pathArch = pathRoot + 'Results/' + arch + '/Metrics/Graphs/'
        pathDSCROC = pathArch + 'DSC_ROC/'
        pathFidelity = pathArch + 'Fidelity_Blur/'
        
        # Load DSC / ROC / Fidelity scores DataFrames
        df_DSC = pd.read_csv(pathDSCROC + 'DSC_Avg.csv', index_col=0).fillna(0)
        df_ROC = pd.read_csv(pathDSCROC + 'ROC_Avg.csv', index_col=0).fillna(0)
        df_Fid = pd.read_csv(pathFidelity + 'Fidelity_Blur_Avg.csv', index_col=0).fillna(0)
        
        # Keep only the specified map types
        df_DSC_sort = df_DSC.loc[:, df_DSC.columns.isin(MAP_TYPES)]
        df_ROC_sort = df_ROC.loc[:, df_ROC.columns.isin(MAP_TYPES)]
        df_Fid_sort = df_Fid.loc[:, df_Fid.columns.isin(MAP_TYPES)]
        
        # Convert each in usable format & shape
        df_DSC_reshape = reshape_DF(df_DSC_sort, 'DSC')
        df_ROC_reshape = reshape_DF(df_ROC_sort, 'ROC')
        df_Fid_reshape = reshape_DF(df_Fid_sort, 'Fidelity')
        
        # Combine all DataFrames
        df_all = pd.concat([df_DSC_reshape, df_ROC_reshape, df_Fid_reshape], axis=1)
        df_all['Method'] = [i[:i.find('_')] for i in df_all.index.to_list()]
        df_all['Map'] = [i[i.find('_')+1:] for i in df_all.index.to_list()]
        
        # Replace long names
        df_all = df_all.replace('Avg(BP.IG(0).GradCAM.EG)', 'Avg(4M)')
        df_all = df_all.replace('Product(BP.IG(0).GradCAM.EG)', 'Product(4M)')
        df_all = df_all.replace('Avg(IG(0).GradCAM)', 'Avg(2M)')
        df_all = df_all.replace('Product(IG(0).GradCAM)', 'Product(2M)')
        
        # # Call Pareto front function
        # pareto_front_DSC_Fid = pareto(df_all, 'DSC', 'Fidelity')
        # df_all['Pareto_DSC_Fid'] = [ele in pareto_front_DSC_Fid for ele in df_all.index.to_list()]
        # pareto_front_ROC_Fid = pareto(df_all, 'ROC', 'Fidelity')
        # df_all['Pareto_ROC_Fid'] = [ele in pareto_front_ROC_Fid for ele in df_all.index.to_list()]
        
        # Call Euclidean distance function
        df_all['Euclid_DSC_Fid'] = euclid_dist(df_all, 'DSC', 'Fidelity', best_point=((1.0, 0.0)))
        df_all['Euclid_ROC_Fid'] = euclid_dist(df_all, 'ROC', 'Fidelity', best_point=((1.0, 0.0)))
        
        # Call Manhattan distance function
        df_all['Manhat_DSC_Fid'] = manhat_dist(df_all, 'DSC', 'Fidelity', best_point=((1.0, 0.0)))
        df_all['Manhat_ROC_Fid'] = manhat_dist(df_all, 'ROC', 'Fidelity', best_point=((1.0, 0.0)))
        
        # Init figure
        figVS, axVS = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), squeeze=False)
        
        # Plot DSC vs Fidelity
        sns.scatterplot(data=df_all, x="DSC", y="Fidelity", hue="Map", style="Method", ax=axVS[0][0], legend='full', s=400).legend(loc='center right', title='DSC vs Fidelity', fontsize='15', title_fontsize='18', bbox_to_anchor= (1.4,0.5))
        # Plot ROC vs Fidelity
        sns.scatterplot(data=df_all, x="ROC", y="Fidelity", hue="Map", style="Method", ax=axVS[0][1], legend='full', s=400).legend(loc='center right', title='ROC vs Fidelity', fontsize='15', title_fontsize='18', bbox_to_anchor= (1.4,0.5))
        
        # Save overall results file
        pathSave = pathArch + 'ExpectationVsFaithfulness/'
        os.makedirs(pathSave, exist_ok=True)
        df_all.to_csv(pathSave + 'all_distances.csv')
        
        # Save curves
        figVS.tight_layout()
        figVS.savefig(pathSave + 'Expect_vs_faith.tiff')




# Using the special variable
if __name__=="__main__": 
    main()