# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:46:44 2023

@author: Wistan
"""


import numpy as np
import torch
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms
from skimage.metrics import mean_squared_error



#------------------------------     Parameters     ------------------------------#



# Base paths
pathRoot = './'


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# List of perturbation versions with max blur level
versions = {
            "V1": 12,
            "V2": 80,
            "V3": 80,
            "V4": 80,
            "V5": 80,
            "V6": 80,
            "V7": 80,
            "V8": 12,
            "V9": 12,
            "V10": 12,
            "V11": 12,
            "V12": 12,
            "V13": 12,
            }


# Add scaling (to respect the original data range)
do_scaling = True
# Add histogram matching (to respect the original data range)
do_hist_match = True


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
                ]


# Name of classes
classes = ['Without', 'With']
# Uncertainty ideal score
uncertain_ideal = 1 / len(classes)
# Uncertainty limit (for perturbation sorting) above or below uncertain_ideal
uncertain_limit = 0.1


# List of data saved
columns_list = classes + ['Uncert_score', 'MSE', 'MSE/Range', 'Perturb_score', 'Version', 'Sigma', 'Min', 'Max']


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#------------------------------------------------------------------------------#
    
#----------------------------     Utilitaries     -----------------------------#



# Compute uncertainty score
def uncertainty(preds):
    uncertain_score = np.mean(np.abs([uncertain_ideal - val for val in preds]))
    return uncertain_score


# Mean pixel value of each class (without & with)
def mean_value_class(test_images, test_labels):
    
    # Init lists of mean values for each class
    listMeansWith = []
    listMeansWithout = []
    # Init mean of each input
    meanInputs = []
    
    # Compute mean of each class
    for sliceIdx, in_tensor in enumerate(test_images):
        
        # Convert original image to numpy & get label
        in_numpy = in_tensor.squeeze().cpu().numpy()
        label = test_labels[sliceIdx]
        
        # Mean value of input image
        mean_tmp = np.mean(in_numpy)
        meanInputs.append(mean_tmp)
        
        if (sliceIdx not in excluded_idx):
            # Add mean of slice to corresponding class avg
            if (label == 1): listMeansWith.append(mean_tmp)
            else: listMeansWithout.append(mean_tmp)
            
    # Mean value of each class
    meanValueWith = np.mean(listMeansWith)
    meanValueWithout = np.mean(listMeansWithout)
    
    return meanInputs, np.abs(meanValueWithout), np.abs(meanValueWith)



# Save perturbed input as .tiff
def save_data(data, filePath, fileName):
    plt.imsave(filePath + fileName + '.tiff', data, cmap='gray')
    plt.clf()



#----------------------------------------------------------------------------------------#
    
#----------------------------     Perturbation Versions     -----------------------------#


def perturb_versions(in_numpy, version, sig, mean_in, meanValueWithout, meanValueWith, label):

# V1 : simple blur
    if (version == "V1"):
        in_perturb = gaussian_filter(in_numpy, sigma=sig)
        return in_perturb

# V2 : blur x input
    if (version == "V2"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp * in_numpy
        return in_perturb

# V3 : blur x mean of input
    if (version == "V3"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp * mean_in
        return in_perturb

# V4 : blur / mean of input & normalize
    if (version == "V4"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_tmp_2 = in_tmp / mean_in
        in_perturb = in_tmp_2 / np.max(in_tmp_2)
        return in_perturb

# V5 : blur x (1 - mean of input)
    if (version == "V5"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp * (1 - mean_in)
        return in_perturb

# V6 : blur x mean of class
    if (version == "V6"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        if (label == 1): in_perturb = in_tmp * meanValueWith
        else: in_perturb = in_tmp * meanValueWithout
        return in_perturb

# V7 : blur x mean of opposite class
    if (version == "V7"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        if (label == 1): in_perturb = in_tmp * meanValueWithout
        else: in_perturb = in_tmp * meanValueWith
        return in_perturb

# V8 : x input then blur
    if (version == "V8"):
        in_prod = in_numpy * in_numpy
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V9 : x (mean of input) then blur
    if (version == "V9"):
        in_prod = in_numpy * mean_in
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V10 : / (mean of input) & normalize then blur
    if (version == "V10"):
        in_tmp = in_numpy / mean_in
        in_prod = in_tmp / np.max(in_tmp)
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V11 : x (1 - mean of input) then blur
    if (version == "V11"):
        in_prod = in_numpy * (1 - mean_in)
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V12 : x (mean of class) then blur
    if (version == "V12"):
        if (label == 1): in_prod = in_numpy * meanValueWith
        else: in_prod = in_numpy * meanValueWithout
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V13 : x (mean of opposite class) then blur
    if (version == "V13"):
        if (label == 1): in_prod = in_numpy * meanValueWithout
        else: in_prod = in_numpy * meanValueWith
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb


#------------------------------------------------------------------------------------------------#
    
#-----------------------     Test blurring versions (per input data)     ------------------------#



def blur_test():
    
    print("Finding best Perturbation")

    # Output paths for DataFrames & selected perturbations
    output_DF = pathRoot + 'Results/Blur_Perturbation/'
    output_select = pathRoot + 'dataset_290/data_perturbed/'
    
    # Test set images & labels
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    test_labels = torch.load(pathRoot + 'test_labels_260.pt').numpy().astype(int)
    
    # Selected perturbations path
    pathSelect = output_select
    os.makedirs(pathSelect, exist_ok=True)
    
    # Mean pixel value of each class (without then with)
    meanInputs, meanValueWithout, meanValueWith = mean_value_class(test_images, test_labels)
    
    # For each network
    for arch in networks:
            
        print("\t Network " + arch)
        
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # Perturbations evaluation results path
        pathDF = output_DF + arch + '/'
        os.makedirs(pathDF, exist_ok=True)
        
        # Overall best perturbation for each input data
        perturbs = []
        
        # Init tensor of blurred images
        blur_images = torch.Tensor()
        
        # For each slice
        for sliceIdx, in_tensor in enumerate(test_images):
            
            # Init best perturbation image
            im_best_perturb = torch.zeros(size=in_tensor.unsqueeze(0).shape, dtype=in_tensor.dtype)
            
            if (sliceIdx not in excluded_idx):
                
                # Display slice number
                print("\t\t Slice", sliceIdx)
                
                # Init all data array
                allScores = []
                
                # Numpy / label / inference
                in_numpy = in_tensor.squeeze().cpu().numpy()
                label = test_labels[sliceIdx]
                pred_in = float(torch.sigmoid(model(in_tensor.unsqueeze(0))).detach().cpu().numpy())
                pred_in_uncert = uncertainty([pred_in, 1-pred_in])
                in_range = np.max(in_numpy) - np.min(in_numpy)
                
                # Data of input
                in_data = {
                                "Image" : in_numpy,
                                "Uncert_score" : pred_in_uncert,
                                "Range" : in_range,
                              }
                
                # Mean value of input image
                mean_in = meanInputs[sliceIdx]
                
                # Init best perturbation data
                data_best_perturb = pd.Series(data=[1000]*len(columns_list),
                                          index=columns_list)

                # For each version
                for version in versions:
                    
                    # Generate list of blur levels
                    sigmas = [n+1 for n in range(versions[version])]
                    
                    # For each std value
                    for idx, sig in enumerate(sigmas):
                        
                        # Compute perturbation
                        in_perturb = perturb_versions(in_numpy, version, sig, mean_in, meanValueWithout, meanValueWith, label)
                        in_perturb = in_perturb.astype(in_numpy.dtype)
                        
                        # Params of perturbation
                        perturb_params = {"Version" : version+'o', "Sigma" : sig}
                        # Evaluate perturbation & compare with best one
                        data_best_perturb, im_best_perturb, perturb_data = blur_evaluate(model, in_perturb, perturb_params, in_data, data_best_perturb, im_best_perturb)
                        
                        # Save in result array
                        allScores.append(perturb_data)
                        
                        
                        # Scaling if needed
                        if do_scaling:
                            scale_range = (np.min(in_numpy), np.max(in_numpy))
                            data_range = (np.min(in_perturb), np.max(in_perturb))
                            in_scale = ((in_perturb - data_range[0]) / (data_range[1] - data_range[0])) * (scale_range[1] - scale_range[0]) + scale_range[0]
                            in_scale = in_scale.astype(in_numpy.dtype)
                            
                            # Params of perturbation
                            perturb_params["Version"] = version+'s'
                            # Evaluate perturbation & compare with best one
                            data_best_perturb, im_best_perturb, perturb_data = blur_evaluate(model, in_scale, perturb_params, in_data, data_best_perturb, im_best_perturb)
                            
                            # Save in result array
                            allScores.append(perturb_data)
                            
                        
                        # Histogram matching if needed
                        if do_hist_match:
                            in_match = match_histograms(in_perturb, in_numpy).astype(in_numpy.dtype)
                            
                            # Params of perturbation
                            perturb_params["Version"] = version+'m'
                            # Evaluate perturbation & compare with best one
                            data_best_perturb, im_best_perturb, perturb_data = blur_evaluate(model, in_match, perturb_params, in_data, data_best_perturb, im_best_perturb)
                            
                            # Save in result array
                            allScores.append(perturb_data)
                        
                        
                # Save all scores results
                df_all_scores = pd.DataFrame(data=allScores, columns=columns_list)
                df_all_scores = df_all_scores.sort_values(by=['Uncert_score'])
                df_all_scores.to_csv(pathDF + 'all_scores_' + str(sliceIdx) + '.csv')
                
                # Save best perturbation with ID / label / original pred
                perturb_tmp = [sliceIdx, label, 1-pred_in, pred_in] + data_best_perturb.to_list() + [np.min(in_numpy), np.max(in_numpy), in_range]
                perturbs.append(perturb_tmp)
            
            # Save best perturbation image
            blur_images = torch.cat((blur_images, im_best_perturb.cpu()), 0)
            
        # Save all selected perturbations images
        torch.save(blur_images, pathSelect + 'blur_slices_' + arch + '.pt')
        
        # Save all selected perturbations scores
        df_results = pd.DataFrame(data=perturbs, columns=['Slice', 'Label', 'Without (Original)', 'With (Original)'] + columns_list + ['MinInput', 'MaxInput', 'InputRange'])
        df_results.to_csv(pathDF + 'Perturbation_Per_Input.csv')



#------------------------------------------------------------------------------------------------#
    
#------------------------     Inference & Similarity on Perturbation     ------------------------#



def blur_evaluate(model, in_perturb, perturb_params, in_data, data_best_perturb, im_best_perturb):
    
    # To tensor + Inference of perturbed image
    in_perturb_tensor = torch.from_numpy(in_perturb).unsqueeze(0).unsqueeze(0).to(device).float()
    pred = float(torch.sigmoid(model(in_perturb_tensor)).detach().cpu().numpy())
    pred_uncert = uncertainty([pred, 1-pred])
    
    # Mean Squared Error (between original & perturbed)
    mse = mean_squared_error(in_data["Image"], in_perturb)
    # MSE in % of input range
    mse_range = mse / in_data["Range"]
    
    # Combined score uncertainty_score / MSE
    perturb_score = pred_uncert + 5*mse_range
    
    # Perturbation data as list for global save
    perturb_data = [1-pred, pred, pred_uncert, mse, mse_range, perturb_score, perturb_params["Version"], perturb_params["Sigma"], np.min(in_perturb), np.max(in_perturb)]
    
    
    # Is best perturbation if pred_uncert < uncert_limit & lower perturb_score
    if ((pred_uncert < min(uncertain_limit, in_data["Uncert_score"]) and perturb_score < data_best_perturb.loc['Perturb_score'])
    or (data_best_perturb.loc['Uncert_score'] > uncertain_limit and pred_uncert < data_best_perturb.loc['Uncert_score'])):
        # Update best perturbation data
        data_best_perturb.update(pd.Series(perturb_data, index=columns_list))
        im_best_perturb = in_perturb_tensor
    
    
    # Return updated data
    return data_best_perturb, im_best_perturb, perturb_data



#------------------------------------------------------------------------------------------------------------#
    
#--------------------------     Generate perturbation dataset (per input data)     --------------------------#



def blur_generate():
    
    print("Generating selected perturbations")
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').to(device)
    test_labels = torch.load(pathRoot + 'test_labels_260.pt').numpy().astype(int)
    
    # Output for blurred dataset
    output = pathRoot + 'dataset_290/data_perturbed/'
    
    # Mean pixel value of each class (without then with)
    meanInputs, meanValueWithout, meanValueWith = mean_value_class(test_images, test_labels)
    
    # For each network
    for arch in networks:
            
        print("\t Network " + arch)
        
        # Saving path
        pathSave = output + arch + '_generated/'
        os.makedirs(pathSave, exist_ok=True)
    
        # Load list of perturbations to apply
        pathDF = pathRoot + 'Results/Blur_Perturbation/' + arch + '/'
        df_perturbations = pd.read_csv(pathDF + 'Perturbation_Per_Input.csv', index_col=0).fillna(0)
        
        # Get lists of perturbated inputs
        slices_list = list(np.unique(df_perturbations['Slice'].to_list()))
        
        # For each slice
        for sliceIdx in slices_list:
            
            # Extract data corresponding to ID
            perturb_data = df_perturbations.loc[df_perturbations['Slice'] == sliceIdx]
        
            # Extract blur parameters
            full_version = perturb_data['Version'].to_list()[0]
            blur_version = full_version[:-1]
            modifier = full_version[-1]
            blur_level = int(perturb_data['Sigma'])
            
            # Input & label
            in_numpy = test_images[sliceIdx].squeeze().cpu().numpy()
            label = test_labels[sliceIdx]
            
            # Mean value of input image
            mean_in = meanInputs[sliceIdx]
            
            # Process perturbation
            in_perturb = perturb_versions(in_numpy, blur_version, blur_level, mean_in, meanValueWithout, meanValueWith, label)
            
            # Scaling if needed
            if modifier == 's':
                scale_range = (np.min(in_numpy), np.max(in_numpy))
                data_range = (np.min(in_perturb), np.max(in_perturb))
                out = ((in_perturb - data_range[0]) / (data_range[1] - data_range[0])) * (scale_range[1] - scale_range[0]) + scale_range[0]
            
            # Histogram matching if needed
            elif modifier == 'm':
                out = match_histograms(in_perturb, in_numpy)
                
            # No modifier
            elif modifier == 'o':
                out = in_perturb
            
            # Keep the same dtype
            out = out.astype(in_numpy.dtype)
            
            # Save perturbed image
            save_data(out, pathSave, 'slice_'+str(sliceIdx)+'_'+full_version+'_'+str(blur_level)+'.tiff')



#---------------------------------------------------------------------------------------#
    
#-----------------------     Concatenate all detailed files     ------------------------#



def concatDetailed():
    
    print("Concatenation of all files")
    
    # Input path for all perturbations
    input_DF = pathRoot + 'Results/Blur_Perturbation/'
    
    # For each network
    for arch in networks:
            
        print("\t Network " + arch)
        
        # Perturbations path for arch
        pathDF = input_DF + arch + '/'
        
        # Empty list for all results
        df_all_results = pd.DataFrame()
        
        # Get detailed scores filenames
        detail_files = [f for f in os.listdir(pathDF) if 'all_scores_' in f]
        
        # For each file in directory
        for name in detail_files:
        
            # Load scores dataframe
            df_scores = pd.read_csv(pathDF + name, index_col=0).fillna(0)
            # Add Image ID in DataFrame
            im_ID = name[name.rfind('_')+1 : name.rfind('.')]
            df_scores.insert(0, "ID", [im_ID]*len(df_scores), True)
        
            # Append this DataFrame to the overall DataFrame
            df_all_results = df_all_results.append(df_scores, ignore_index=True)
            
        # Save overall results file
        df_all_results.to_csv(pathDF + 'All_Perturbations.csv')
    


#-----------------------------------------------------------------------------------#
    
#----------------------------------     MAIN     -----------------------------------#



# Using the special variable
if __name__=="__main__": 
    
    # Execute different blurring versions and select best one
    blur_test()
    
    # Concatenation of all perturbations for all images
    concatDetailed()
    
    # Generate blurring selection from input file (optional)
    blur_generate()