# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""



import os
import torch
import numpy as np
import pandas as pd
import pickle
import shap



# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Raw_attrs/'
methodFolder = 'EG/'
runsFolder = methodFolder + 'Runs/'


# Number of EG runs & nb of samples
nbRuns = 15
nb_samples = 200


# Convert to absolute if needed
do_absolute = True


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Defining main function
def main(): 
    
    # Background slices load
    background = torch.load(pathRoot + 'baseline_200.pt').to(device)
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_263.pt').to(device)
    # Load labels dataframe
    test_labels = pd.read_csv(pathRoot + 'test_labels_263.csv', index_col=0)
    
    # Load model
    model = pickle.load(open(pathRoot + 'serialized_param_predictor.sav', 'rb')).to(device)
    model.eval()
    
    # Load params dictionnary
    f = open("./params.pkl", "rb")
    params = pickle.load(f)
    f.close()
    
    # GradientExplainer
    e = shap.GradientExplainer(model, background)
    
    # For each test slice
    for sliceIdx in range (len(test_images)):
            
        print("EG Image number", sliceIdx+1)
        
        # Load input image and raw attributions for given slice and method
        im_tensor = test_images[sliceIdx].unsqueeze(0).to(device)
        
        # Retrieve output from the image
        y_pred = model(im_tensor)
    
        # For each parameter prediction
        for param_name, pred in y_pred.items():
            
            print("\t Param", param_name)
            
            # Path for runs & final results
            pathAttr = pathResults + param_name + '/' + methodFolder
            pathRuns = pathResults + param_name + '/' + runsFolder
            os.makedirs(pathRuns, exist_ok=True) 
            
            # Apply final layer (softmax or sigmoid depending on parameter)
            pred_class = 0
            if (param_name != 'contrast_used'): pred_class = torch.sigmoid(pred)
            
            # Extract specific label
            label = test_labels[param_name].tolist()[sliceIdx]
            
            # Empty list for storing all runs results
            attr_all_runs = []
    
            # For each run (random seed)
            for rseed in range (nbRuns):
                print("\t\t Run number", rseed+1)
            
                # Compute SHAP values for given examples
                attr, indexes = e.shap_values(im_tensor, nsamples=nb_samples, ranked_outputs=1, rseed=rseed, target_branch=param_name)
                attr = attr[0].squeeze()
                
                # Get predicted class
                indexes = indexes.detach().cpu().numpy()[0][0]
                pred_class = params[param_name][indexes]
                
                # Save raw array for each slice for each run
                if (param_name == 'contrast_used'):
                    # Invert attribution values if label is "without contrast agent"
                    if (label == False): attr = attr * -1
                
                # Save run map
                np.save(pathRuns + 'EG_Im_' +str(sliceIdx)+ '_Run_' +str(rseed) + '_pred_' + str(pred_class) + '.npy', attr)
            
                # Append run to list
                attr_all_runs.append(attr)
    
            # Mean of all runs
            attr_avg = np.mean(attr_all_runs, axis=0)
            
            # Save average map
            np.save(pathAttr + 'EG_Im_' +str(sliceIdx)+ '_pred_' + str(pred_class) + '_label_' + str(label) + '.npy', attr_avg)
            
            # Convert to absolute if needed
            if do_absolute:
                attr_avg_abs = np.abs(attr_avg)
                pathAttr_abs = pathAttr.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
                os.makedirs(pathAttr_abs, exist_ok=True)
                np.save(pathAttr_abs + 'EG_Im_' +str(sliceIdx)+ '_pred_' + str(pred_class) + '_label_' + str(label) + '.npy', attr_avg_abs)
            


# Using the special variable
if __name__=="__main__": 
    main()