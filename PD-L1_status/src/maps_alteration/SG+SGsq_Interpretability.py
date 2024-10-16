# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:55:04 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import pandas as pd

from XAI_SG_BP import BP
from XAI_SG_Deconv import Deconv
from XAI_SG_GradCAM import GradCAM
from XAI_SG_IG import IG
from XAI_SG_EG import EG

from utils.XAI_utils import XAI_dataset
from utils.utils import replace_layers
from utils.manage_immugast import immugast_split
from utils.train_utils import get_transforms
from utils.exam_immugast import exam_immugast


#-----------------------------     PARAMETERS     -----------------------------#


# Base paths
loading = './'
save_base = loading + 'Results/'


# List of folds
trials = {
            0 : 'layer4.2.conv3.conv',
            1 : 'layer4.0.conv1.conv',
            2 : 'layer4.0.conv1.conv',
            3 : 'layer4.1.conv1.conv',
            4 : 'layer4.1.conv1.conv',
          }


# All methods to apply
INTERPRET_METHODS = {
                    # 'BP' : BP,
                    # 'Deconv' : Deconv,
                    # 'GradCAM' : GradCAM,
                    # 'IG(Zero)' : IG,
                    'EG' : EG,
                     }


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
             }


# SmoothGrad & SmoothGrad Squared
SG_TYPES = [
            'SG',
            'SGSQ',
            ]


# Samples Number Parameter
nsamples = 50


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


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#--------------------------------     MAIN     --------------------------------#


# Defining main function
def main():
    
    # For each trial (=fold)
    for trial in trials:
        
        print('Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, test_loader, net = XAI_dataset(params, trial, device, loading)
        
        # Replace inplace of ReLU to False
        replace_layers(net)
        
        # Load list of best noise levels for SG & SG²
        df_noise_lvls = pd.read_csv(save_base + 't-' + str(trial) + '_SG_SGsq_noise_lvls.csv', index_col=0).fillna(0)
            
        # For each interpretability method
        for method in INTERPRET_METHODS:
            
            print('\t', method)
            
            # Extract corresponding noise levels
            df_noise_method = df_noise_lvls[df_noise_lvls['Method'] == method]
            
            # If SG or SG² is used
            for SG_type in SG_TYPES:
                
                print('\t\t' + SG_type)
                    
                # Extract corresponding noise levels
                df_noise_SG = df_noise_method[df_noise_method['SG-SGSQ'] == SG_type]
                
                # For each map type
                for map_type in MAP_TYPES:
                    
                    print('\t\t\t' + map_type)
                    
                    # Path for saving results
                    methodNewName = SG_type + '+' + method
                    pathResults = save_base + MAP_TYPES[map_type] + '/' + methodNewName + '/' + trial_folder
                    os.makedirs(pathResults, exist_ok=True)
                    
                    # Extract noise level to apply
                    noise_lvl = df_noise_SG[map_type].to_list()[0]
                    
                    # Check if noise level is != than 0.0
                    if (noise_lvl != 0.0):
                        
                        # Add method-specific parameters
                        GradCAM_params = {}
                        IG_params = {}
                        EG_params = {}
                        
                        if ('IG' in method):
                            # IG parameters
                            IG_params = {
                                            'SaveBase' : save_base + MAP_TYPES[map_type] + '/' + SG_type + '+' + method + '/',
                                        }
                        elif (method == 'GradCAM'):
                            # Identify GradCAM layer
                            for i, d in enumerate(net.named_modules()):
                                # If selected layer
                                if (d[0] == trials[trial]):
                                    # Save layer and break for loop
                                    conv_layer = d[1]
                                    break
                            # GradCAM parameters
                            GradCAM_params = {
                                                'Layer' : conv_layer,
                                             }
                        elif (method == 'EG'):
                            # IDs of train patients
                            train_ids, _, _ = immugast_split(params['feature'], params['cutoff'], trial)
                            # Train patients transforms
                            train_transforms, _, _ = get_transforms('immugast', params['size'])
                            # Iterate on train subset
                            train_imgs, train_labels = list(), list()
                            for train_id in train_ids:
                                exam = exam_immugast(int(train_id), loading, upload=False, offsetx=params['offsetx'], offsety=params['offsety'], offsetz=params['offsetz'])
                                train_imgs.append(exam.folder+exam.id+'.nii.gz')
                                if params['feature'] == 'CPS':
                                    exam.binarize_CPS(params['cutoff'])
                                    train_labels.append(int(exam.CPSb))  
                            train_labels = np.array(train_labels, dtype=int)
                            # EG background parameters
                            EG_params = {
                                            'Transforms' : train_transforms,
                                            'Imgs' : train_imgs,
                                            'Labels' : train_labels,
                                        }
                        
                        # For each element in test loader
                        for step, data in enumerate(test_loader):
                            
                            # Display id
                            print('\t\t\t\t' + ids[step])
                            
                            # Load original input
                            in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
                            in_numpy = in_tensor.squeeze().cpu().numpy()
                            
                            # Standard Deviation of the noise
                            stdev = noise_lvl * (np.max(in_numpy) - np.min(in_numpy))
                            
                            # Array of all noisy samples
                            all_samples = []
                            
                            # For each sample
                            for i in range(nsamples):
                                # Apply noise to the input image
                                noise = np.random.normal(0, stdev, in_numpy.shape)
                                x_plus_noise = in_numpy + noise
                                # Add sample to array
                                all_samples.append(x_plus_noise)
                            
                            # Convert to array
                            all_samples = np.array(all_samples)
                            # Convert to tensor
                            all_samples_tensor = torch.tensor(all_samples, device=device, dtype=torch.float32).unsqueeze(1)
                            
                            # Input parameters
                            input_params = {
                                            'Network' : net,
                                            'Tensor' : all_samples_tensor,
                                            'Numpy' : all_samples,
                                            'Label' : label,
                                            'Trial' : trial_folder,
                                            'Absolute' : True if ('abs' in map_type) else False,
                                            'SG_type' : SG_type,
                                            'FilePrefix' : MAP_TYPES[map_type] + '_' + methodNewName,
                                           }
                            
                            # Get the original file name and affine
                            original_filepath = data[0].meta['filename_or_obj'][0]
                            original_id = original_filepath[original_filepath.rfind('\\')+1 : original_filepath.rfind('.')]
                            affine = data[0].meta['affine'].squeeze()
                            
                            # Output parameters
                            output_params = {
                                                'ID' : original_id,
                                                'Savepath' : pathResults,
                                                'Affine' : affine,
                                            }
                            
                            # Call method
                            INTERPRET_METHODS[method](input_params, output_params, EG_params, GradCAM_params, IG_params)


# Using the special variable
if __name__=="__main__": 
    main()