# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import numpy as np
import shap
import cv2
import tensorflow as tf


# Base paths
pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathBackground = pathRoot + 'Background Images/'
pathResults = pathRoot + 'Results/Raw_attrs/ExpectedGradients/'
pathResultsRuns = pathResults + 'Runs/'
pathRawAttr = pathRoot + 'Results/Methods_Attributions_Inversed/'

# Define device
device = tf.device('cpu')

# Load model
model = tf.keras.models.load_model(pathRoot + 'serialized_model', compile=False)

# Declare number of runs
nbRuns = 15

# Defining main function
def main(): 

    # Get all subdirectories
    list_all_dirs_files = [x for x in os.walk(pathImages)]
    dirs = list_all_dirs_files[0][1]
    
    # Images chosen for baseline of saliency maps
    bkgd_paths = [f for f in os.listdir(pathBackground) if os.path.isfile(os.path.join(pathBackground, f)) and f.endswith('.tif')]
    background = [cv2.imread(pathBackground + bkgd_paths[i]) for i in range(len(bkgd_paths))]
    
    # GradientExplainer
    print("Start Explainer...")
    e = shap.GradientExplainer(model, background)
    print("End Explainer...")
    
    # For each subdirectory
    for dirIdx in range(len(dirs)):
        
        dir_path = pathImages + dirs[dirIdx] + '/'
        
        print("Directory :", dir_path)
    
        # Images chosen for application of saliency maps
        images_paths = list_all_dirs_files[dirIdx+1][2]
    
        # Images chosen for application of saliency maps
        test_images = [cv2.imread(dir_path + images_paths[i]).astype('uint8') / 255 for i in range(len(images_paths))]
        test_images = np.array(test_images)
        
        # Empty list for storing all runs results
        raw_all_runs = []
        
        # For each run (random seed)
        for rseed in range (nbRuns):
            print("\t Expected Gradients Run number", rseed+1)
        
            # Compute SHAP values for given examples
            with device:
                print("\t\t Start Application...")
                shap_values = e.shap_values(test_images, rseed=rseed)[0]
                print("\t\t End Application...")
            
            # For each test slice
            for imageIdx in range (len(test_images)):
                
                # Save raw array for each slice for each run
                np.save(pathResultsRuns + 'Raw_ExpectedGradients_' +images_paths[imageIdx]+ '_Run_' +str(rseed+1)+ '.npy', shap_values[imageIdx])
        
            # Append run to list
            raw_all_runs.append(shap_values)
        
        # Mean of all runs
        raw_mean = np.mean(raw_all_runs, axis=0)
        
        # For each test slice
        for imageIdx in range (len(raw_mean)):
            # Save average arrays, one by test image
            raw_max = np.max(raw_mean[imageIdx], axis=-1)
            
            np.save(pathResults + dirs[dirIdx] + '/Raw_ExpectedGradients_' +images_paths[imageIdx]+ '.npy', raw_max)



# Using the special variable
if __name__=="__main__": 
    main()