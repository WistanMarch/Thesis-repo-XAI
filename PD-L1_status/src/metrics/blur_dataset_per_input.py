# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:46:44 2023

@author: Wistan
"""


from utils.XAI_utils import XAI_dataset
from utils.utils import mean_value_class
import numpy as np
import torch
import torch.nn as nn
import os

import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms
from skimage.metrics import mean_squared_error



#------------------------------     Parameters     ------------------------------#



# Base paths
loading = './'


# List of folds
trials = [
            0,
            1,
            2,
            3,
            4,
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
            "V14": 80,
            "V15": 12,
            }


# Blur std values
sigmas = [sig for sig in range(1, 61)]


# Add scaling (to respect the original data range)
do_scaling = True
# Add histogram matching (to respect the original data range)
do_hist_match = True


# Name of classes
classes = ['Neg', 'Pos']
# Uncertainty ideal score
uncertain_ideal = 1 / len(classes)
# Uncertainty limit (for perturbation sorting) above or below 1/nb_classes
uncertain_limit = 0.1


# Fixed Parameters (since training is complete)
params = {
            'net_id': 11,
            'feature': 'CPS',
            'cutoff': 1,
            'batch': 1,
            'size': 192,
            'offsetx': 0,
            'offsety': 0,
            'offsetz': 0,
         }


# List of columns saved
columns_list = classes + ['Uncert_score', 'MSE', 'MSE/Range', 'Perturb_score', 'Version', 'Sigma', 'Min', 'Max']


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#------------------------------------------------------------------------------#
    
#----------------------------     Utilitaries     -----------------------------#


# Compute uncertainty score
def uncertainty(preds):
    uncertain_score = np.mean(np.abs([uncertain_ideal - val for val in preds]))
    return uncertain_score


# Save perturbed data as .nii file
def save_data(data, affine, filePath, fileName):
    ni_img = nib.Nifti1Image(data, affine=affine)
    os.makedirs(filePath, exist_ok=True)
    nib.save(ni_img, filePath + fileName)


#----------------------------------------------------------------------------------------#
    
#----------------------------     Perturbation Versions     -----------------------------#


def perturb_versions(in_numpy, version, sig, mean_in, meanValueNeg, meanValuePos, label):

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

# V4 : blur / mean of input
    if (version == "V4"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp / mean_in
        return in_perturb

# V5 : blur x (1 - mean of input)
    if (version == "V5"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp * (1 - mean_in)
        return in_perturb

# V6 : blur x mean of class
    if (version == "V6"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        if (label == 1): in_perturb = in_tmp * meanValuePos
        else: in_perturb = in_tmp * meanValueNeg
        return in_perturb

# V7 : blur x mean of opposite class
    if (version == "V7"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        if (label == 1): in_perturb = in_tmp * meanValueNeg
        else: in_perturb = in_tmp * meanValuePos
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

# V10 : / (mean of input) then blur
    if (version == "V10"):
        in_prod = in_numpy / mean_in
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V11 : x (1 - mean of input) then blur
    if (version == "V11"):
        in_prod = in_numpy * (1 - mean_in)
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V12 : x (mean of class) then blur
    if (version == "V12"):
        if (label == 1): in_prod = in_numpy * meanValuePos
        else: in_prod = in_numpy * meanValueNeg
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V13 : x (mean of opposite class) then blur
    if (version == "V13"):
        if (label == 1): in_prod = in_numpy * meanValueNeg
        else: in_prod = in_numpy * meanValuePos
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb

# V14 : blur x input² (keep sign)
    if (version == "V14"):
        in_tmp = gaussian_filter(in_numpy, sigma=sig)
        in_perturb = in_tmp * in_numpy * in_numpy
        return in_perturb

# V15 : x input² then blur (keep sign)
    if (version == "V15"):
        in_prod = in_numpy * in_numpy * in_numpy
        in_perturb = gaussian_filter(in_prod, sigma=sig)
        return in_perturb


#----------------------------------------------------------------------------------------------------------#
    
#----------------------------     Test blurring versions (per input data)     -----------------------------#



def blur_test():
    
    print("Finding best Perturbation")

    # Output paths for DataFrames & selected perturbations
    output_DF = loading + 'Results/Blur_Perturbation/'
    output_select = loading + 'dataset_crop-x-0-y-0-z-0/data_perturbed/'
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        # Saving path
        pathDF = output_DF + trial_folder
        os.makedirs(pathDF, exist_ok=True)
        pathSelect = output_select + trial_folder
        os.makedirs(pathSelect, exist_ok=True)
    
        # Create dataset for XAI
        ids, labels, loader, net = XAI_dataset(params, trial, device, loading)
        
        # Count inputs for each class
        nbPos = np.count_nonzero(labels)
        nbNeg = len(labels) - nbPos
        
        # Mean pixel value of each class (negative & positive)
        meanInputs, meanValueNeg, meanValuePos = mean_value_class(loader, nbPos, nbNeg)
        
        # Overall best perturbation for each input data
        perturbs = []
        
        # For each element in loader
        for step, data in enumerate(loader):
            
            # Display id
            print("\t\t ID", ids[step])
                
            # Init results arrays
            allPreds = []
            
            # Inference
            in_tensor, label = data[0].to(device), data[1].cpu().numpy()
            in_numpy = in_tensor.squeeze().cpu().numpy()
            affine = in_tensor.meta['affine'].squeeze()
            pred_in = nn.functional.softmax(net(in_tensor),dim=1).detach().cpu().numpy()[0]
            pred_in_uncert = uncertainty(pred_in)
            in_range = np.max(in_numpy) - np.min(in_numpy)
            
            # Data of input
            in_data = {
                            "Image" : in_numpy,
                            "Uncert_score" : pred_in_uncert,
                            "Range" : in_range,
                         }
            
            # Mean value of input image
            mean_in = meanInputs[step]
            
            # Init best perturbation
            best_perturb = pd.Series(data=[1000]*len(columns_list),
                                     index=columns_list)
            out_perturb = np.zeros(shape=in_numpy.shape,
                                   dtype=in_numpy.dtype)
            
            # For each version
            for version in versions:
                
                print("\t\t\t" + version)

                # Generate list of blur levels
                sigmas = [n+1 for n in range(versions[version])]
                
                # For each std value
                for idx, sig in enumerate(sigmas):
            
                    # Process perturbation
                    in_perturb = perturb_versions(in_numpy, version, sig, mean_in, meanValueNeg, meanValuePos, label)
                    in_perturb = in_perturb.astype(in_numpy.dtype)
                    
                    # Params of perturbation
                    perturb_params = {"Version" : version+'o', "Sigma" : sig}
                    # Evaluate processed perturbation & compare with best one
                    best_perturb, out_perturb, perturb_data = blur_evaluate(net, in_perturb, perturb_params, in_data, best_perturb, out_perturb)
                    
                    # Save in result array
                    allPreds.append(perturb_data)
                    
                    
                    # Scaling if needed
                    if do_scaling:
                        scale_range = (np.min(in_numpy), np.max(in_numpy))
                        data_range = (np.min(in_perturb), np.max(in_perturb))
                        in_scale = ((in_perturb - data_range[0]) / (data_range[1] - data_range[0])) * (scale_range[1] - scale_range[0]) + scale_range[0]
                        in_scale = in_scale.astype(in_numpy.dtype)
                        
                        # Params of perturbation
                        perturb_params["Version"] = version+'s'
                        # Evaluate processed perturbation & compare with best one
                        best_perturb, out_perturb, perturb_data = blur_evaluate(net, in_scale, perturb_params, in_data, best_perturb, out_perturb)
                        
                        # Save in result array
                        allPreds.append(perturb_data)
                        
                    
                    # Histogram matching if needed
                    if do_hist_match:
                        in_match = match_histograms(in_perturb, in_numpy)
                        in_match = in_match.astype(in_numpy.dtype)
                        
                        # Params of perturbation
                        perturb_params["Version"] = version+'m'
                        # Evaluate processed perturbation & compare with best one
                        best_perturb, out_perturb, perturb_data = blur_evaluate(net, in_match, perturb_params, in_data, best_perturb, out_perturb)
                        
                        # Save in result array
                        allPreds.append(perturb_data)
                        
                    
            # Save all predictions results
            df_all_preds = pd.DataFrame(data=allPreds, columns=columns_list)
            df_all_preds = df_all_preds.sort_values(by=['Uncert_score'])
            df_all_preds.to_csv(pathDF + 'all_scores_' + ids[step] + '.csv')
            
            # Save best perturbation image
            save_data(out_perturb, affine, filePath=pathSelect, fileName='blur_' + ids[step] + '.nii')
            
            # Save best perturbation data with ID / label / original pred
            perturb_tmp = [ids[step], labels[step], pred_in[0], pred_in[1]] + best_perturb.to_list() + [np.min(in_numpy), np.max(in_numpy), in_range]
            perturbs.append(perturb_tmp)
            
        # Save average results
        df_results = pd.DataFrame(data=perturbs, columns=['ID', 'Label', 'Neg (Original)', 'Pos (Original)'] + columns_list + ['MinInput', 'MaxInput', 'InputRange'])
        df_results.to_csv(pathDF + 'Perturbation_Per_Input.csv')
                        


#------------------------------------------------------------------------------------------------#
    
#-------------------     Inference & Histogram Comparison on blurred data     -------------------#


def blur_evaluate(net, in_perturb, perturb_params, in_data, best_perturb, out_perturb):
    
    # To tensor + Inference of perturbed image
    in_perturb_tensor = torch.from_numpy(in_perturb).unsqueeze(0).unsqueeze(0).to(device).float()
    pred = nn.functional.softmax(net(in_perturb_tensor),dim=1).detach().cpu().numpy()[0]
    pred_uncert = uncertainty(pred)
    
    # Similarity metrics (between original & perturbed)
    mse = mean_squared_error(in_data["Image"], in_perturb)
    # MSE in % of input range
    mse_range = mse / in_data["Range"]
    
    # Combined score with MSE (custom function to weight each part)
    perturb_score = pred_uncert + 5*mse_range
    
    # Perturbation data as list for global save
    perturb_data = [pred[0], pred[1], pred_uncert, mse, mse_range, perturb_score, perturb_params["Version"], perturb_params["Sigma"], np.min(in_perturb), np.max(in_perturb)]
    
    
    # Is best perturbation if diff < 0.2 & lower perturb_score
    if ((pred_uncert < min(uncertain_limit, in_data["Uncert_score"]) and perturb_score < best_perturb.loc['Perturb_score'])
    or (best_perturb.loc['Uncert_score'] > uncertain_limit and pred_uncert < best_perturb.loc['Uncert_score'])):
        # Update best perturbation data
        best_perturb.update(pd.Series(perturb_data, index=columns_list))
        out_perturb = in_perturb
    
    
    # Return updated data
    return best_perturb, out_perturb, perturb_data



#------------------------------------------------------------------------------------------------------------#
    
#----------------------------     Generate blurred dataset (per input data)     -----------------------------#



def blur_generate():
    
    print("Generating selected perturbations")

    # Output for blurred dataset
    output = loading + 'dataset_crop-x-0-y-0-z-0/data_perturbed/'
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Loading & saving paths
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        # Saving path
        pathSave = output + trial_folder
        os.makedirs(pathSave, exist_ok=True)
    
        # Create dataset for XAI
        ids, labels, loader, _ = XAI_dataset(params, trial, device, loading)
        
        # Count inputs for each class
        nbPos = np.count_nonzero(labels)
        nbNeg = len(labels) - nbPos
        
        # Mean pixel value of each class (negative & positive)
        meanInputs, meanValueNeg, meanValuePos = mean_value_class(loader, nbPos, nbNeg)
        
        # Load list of perturbations to apply
        pathDF = loading + 'Results/Blur_Perturbation/' + trial_folder
        os.makedirs(pathDF, exist_ok=True)
        df_perturbations = pd.read_csv(pathDF + 'Perturbation_Per_Input.csv', index_col=0).fillna(0)
        
        # For each element in loader
        for step, data in enumerate(loader):
            
            # Display id
            print("\t\t ID", ids[step])
            
            # Extract data corresponding to ID
            perturb_data = df_perturbations.loc[df_perturbations['ID'] == int(ids[step])]
        
            # Chosen blur parameters
            full_version = perturb_data['Version'].to_list()[0]
            blur_version = full_version[:-1]
            modifier = full_version[-1]
            blur_level = int(perturb_data['Sigma'])
            print("\t\t\t Blur version: " + full_version + ", level: " + str(blur_level))
            
            # Inference
            in_tensor = data[0].to(device)
            label = data[1].cpu().numpy()
            in_numpy = in_tensor.squeeze().cpu().numpy()
            affine = in_tensor.meta['affine'].squeeze()
            
            # Mean value of input image
            mean_in = meanInputs[step]
            
            # Process perturbation
            in_perturb = perturb_versions(in_numpy, blur_version, blur_level, mean_in, meanValueNeg, meanValuePos, label)
            
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
            
            # Save perturbed input as .nii file
            save_data(out, affine, output+trial_folder, 'ID_'+ids[step]+'_'+full_version+'_'+str(blur_level)+'.nii')
            


#---------------------------------------------------------------------------------------#
    
#-----------------------     Concatenate all detailed files     ------------------------#



def concatDetailed():
    
    print("Concatenation of all files")
    
    # Input path for all perturbations
    input_DF = loading + 'Results/Blur_Perturbation/'
    
    # For each trial (=fold)
    for trial in trials:
        
        print('\t Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        # Loading path
        pathDF = input_DF + trial_folder
        
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
    
    # Execute different blurring versions
    blur_test()
    
    # Concatenation of all perturbations for all images
    concatDetailed()
    
    # # Generate blurring selection from input file (optional)
    # blur_generate()