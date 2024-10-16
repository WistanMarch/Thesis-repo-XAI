# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:21:02 2023

@author: Wistan
"""


import logging
import sys
import os
import torch
import numpy as np

from XAI_BP import BP
from XAI_Deconv import Deconv
from XAI_GradCAM import GradCAM
from XAI_IG import IG
from XAI_EG import EG
from XAI_Random import Random

from utils.XAI_utils import XAI_dataset
from utils.utils import replace_layers
from utils.manage_immugast import immugast_split
from utils.train_utils import get_transforms
from utils.exam_immugast import exam_immugast



#------------------------------------     PARAMETERS     ------------------------------------#


# Base paths
loading = './'
save_base = loading + 'Results/Raw_attrs/'


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
                    # 'IG' : IG,
                    # 'EG' : EG,
                    'Random' : Random,
                     }


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


# Also compute Absolute version of attribution map
do_absolute = True


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#-----------------------------------------     Prepare Interpretability     -----------------------------------------#



# Call any implemented Interpretability method
def XAI_immugast_3D():
    
    # For each trial (=fold)
    for trial in trials:
        
        print('Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, test_loader, net = XAI_dataset(params, trial, device, loading)
        
        # Replace inplace of ReLU to False
        replace_layers(net)
        
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
        
        # IG parameters
        IG_params = {
                        'SaveBase' : save_base,
                    }
        
        # For each interpretability method
        for method in INTERPRET_METHODS:
            
            print('\t', method)
            
            # Save folder
            pathSave = save_base + method + '/' + trial_folder
            
            # For each element in test loader
            for step, data in enumerate(test_loader):
                
                # Display id
                print("\t\t ID", ids[step])
                
                # Load original input
                in_tensor, label = data[0].to(device), data[1].squeeze().cpu().numpy()
                in_numpy = in_tensor.squeeze().cpu().numpy()
                
                # Input parameters
                input_params = {
                                'Network' : net,
                                'Tensor' : in_tensor,
                                'Numpy' : in_numpy,
                                'Label' : label,
                                'Trial' : trial_folder,
                                'Absolute' : do_absolute,
                               }
                
                # Get the original file name and affine
                original_filepath = data[0].meta['filename_or_obj'][0]
                original_id = original_filepath[original_filepath.rfind('\\')+1 : original_filepath.rfind('.')]
                affine = data[0].meta['affine'].squeeze()
                
                # Output parameters
                output_params = {
                                    'ID' : original_id,
                                    'Savepath' : pathSave,
                                    'Affine' : affine,
                                }
                
                # Call method
                INTERPRET_METHODS[method](input_params, output_params, EG_params, GradCAM_params, IG_params)



#-----------------------------------------     MAIN     -----------------------------------------#



if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        XAI_immugast_3D()

    except KeyboardInterrupt:
        logging.info('keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
