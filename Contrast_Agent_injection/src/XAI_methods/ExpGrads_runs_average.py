# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import pickle
import shap


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'

nbRuns = 1


# Defining main function
def main(): 
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200_3rd.pt').to(device)
    bkgd_batches = torch.split(background, 40)
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    # Load labels tensor
    test_labels = torch.load(pathRoot + 'test_labels_260.pt').numpy()
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/Raw_attrs/EG/'
        # pathResultsRuns = pathResults + 'Runs/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
            
        # Empty list for storing all runs results
        raw_all_runs = []
        
        # For each run (random seed)
        for rseed in range (nbRuns):
            print("\t Expected Gradients Run number", rseed+1)
            
            # One run attributions
            raw_all_batches = []
        
            # For each batch of test_images
            for batch in bkgd_batches:
                
                # GradientExplainer
                print("\t\t Start Explainer...")
                # e = shap.GradientExplainer(model, background)
                e = shap.GradientExplainer(model, batch)
                print("\t\t End Explainer...")
        
                # # Saving explainer for further loading
                # f = open(pathResults + 'gradientExplainer_260.txt', 'wb')
                # pickle.dump(e, f)
                # f.close()
            
                # # Loading explainer from file
                # f = open(pathResults + 'gradientExplainer_290.txt', 'rb')
                # e = pickle.load(f)
                # f.close()
                # print("Load Explainer OK")
        
                # Compute SHAP values for given examples
                print("\t\t Start Application...")
                shap_values = e.shap_values(test_images, nsamples=len(batch), rseed=rseed).squeeze()
                print("\t\t End Application...")
            
                # For each test slice
                for sliceIdx in range (len(test_images)):
                    # Invert attribution values if label is "without contrast agent"
                    if (test_labels[sliceIdx] <= 0.5): shap_values[sliceIdx] = shap_values[sliceIdx] * -1
                
                # Save as part of one run
                raw_all_batches.append(shap_values)
            
            raw_one_run = np.mean(raw_all_batches, axis=0)
            
            # # For each test slice
            # for sliceIdx in range (len(test_images)):
            #     # Save raw array for each slice for each run
            #     os.makedirs(pathResultsRuns, exist_ok=True)
            #     np.save(pathResultsRuns + 'Raw_attr_ExpectedGradients_Im_' +str(sliceIdx+1)+ '_Run_' +str(rseed+1)+ '.npy', raw_one_run[sliceIdx])
        
            # Append run to list
            raw_all_runs.append(raw_one_run)
        
        # Mean of all runs
        raw_mean = np.mean(raw_all_runs, axis=0)
        
        # For each test slice
        for sliceIdx in range (len(test_images)):
            # Save average arrays, one by test image
            os.makedirs(pathResults, exist_ok=True)
            np.save(pathResults + 'Raw_attr_EG_Im_' +str(sliceIdx+1)+ '.npy', raw_mean[sliceIdx])



# Using the special variable
if __name__=="__main__": 
    main()