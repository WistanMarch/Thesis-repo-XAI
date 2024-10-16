# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:08:29 2021

@author: Wistan
"""



import os
import numpy as np
import torch
import nibabel as nib
import seaborn as sns

from matplotlib import pylab as P

from utils.XAI_utils import XAI_dataset
from utils_display import set_size, ShowImage, ShowHeatMap



#-----------------------------     Parameters     -----------------------------#



# Base paths
loading = './'
pathResults = loading + 'Results/'


# List of folds
trials = [
            # 0,
            # 1,
            2,
            3,
            4,
          ]


# All methods to apply
INTERPRET_METHODS = [
                        'BP',
                        'EG',
                        'GradCAM',
                        'Deconv',
                        'IG(Min)',
                        'IG(Zero)',
                        'IG(Max)',
                        'IG(Zero-Min)',
                        'IG(Min-Max)',
                        'IG(Zero-Max)',
                        'IG(Avg)',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
            # 'reg_abs' : 'XRAI_attrs(absolute)',
            # 'reg_orig' : 'XRAI_attrs',
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


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set up matplotlib figures
ROWS = 1
COLS = len(INTERPRET_METHODS) + 1
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72


# Apply custom theme
palette = sns.color_palette("bright")
sns.set_theme(context="poster", style='whitegrid', palette=palette, font_scale=0.7)



#------------------------------------------------------------------------------#

#--------------------------------     UTILS     -------------------------------#



def transforms_image(im):
    im_transf = np.transpose(np.flip(im))
    return im_transf



#------------------------------------------------------------------------------#

#-----------------------------     MAPS DISPLAY     ---------------------------#



def maps_multi_display():
    
    # For each trial (=fold)
    for trial in trials:
        
        print('Trial', trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, _ = XAI_dataset(params, trial, device, loading)
        
        # For each map type
        for map_type in MAP_TYPES:
            
            print("\t Map Type", map_type)
            
            # For each element in loader
            for step, data in enumerate(loader):
                
                # Display id
                print("\t\t ID", ids[step])
                
                # Load original input / Inference
                in_tensor, _ = data[0], data[1].squeeze().cpu().numpy()
                in_numpy = in_tensor.squeeze().cpu().numpy()
                
                # List of all methods maps
                methods_maps = []
                
                # For each method
                for method in (INTERPRET_METHODS):
                    
                    # Path for corresponding attribution maps
                    pathLoadFolder = pathResults + MAP_TYPES[map_type] + '/' + method + '/' + trial_folder
                    
                    # Load attribution map
                    ni_map_attr = nib.load(pathLoadFolder + MAP_TYPES[map_type] + '_' + method + '_' + ids[step] + '.nii')
                    map_attr = ni_map_attr.get_fdata()
                    methods_maps.append(map_attr)
                    
                # For every slice of the input image
                for slice_idx in range(in_numpy.shape[-1]):
                    
                    im_slice = transforms_image(in_numpy[:,:,slice_idx])
                    
                    P.figure(figsize=(1, 1), dpi=DPI)
                    # Display percentages
                    set_size(im_slice.shape[1]*UPSCALE_FACTOR/DPI, (im_slice.shape[0]/DPI)+1)
                    
                    # Show original image
                    ShowImage(im_slice, title='Input Slice n°'+str(slice_idx+1), ax=P.subplot(ROWS, COLS, 1))
                    
                    # Display corresponding slice for each method
                    for method_idx, method_map in enumerate(methods_maps):
                        
                        method_slice = transforms_image(method_map[:,:,slice_idx])
                        # Show method slice
                        ShowHeatMap(method_slice, title='Method '+INTERPRET_METHODS[method_idx], ax=P.subplot(ROWS, COLS, method_idx+2))
                        
                    # Save image
                    pathSave = pathResults + 'Multi-Display/' + trial_folder + ids[step] + '/'
                    os.makedirs(pathSave, exist_ok=True)
                    P.savefig(pathSave + ids[step] + '_' + str(slice_idx+1) + '.tiff')
                    P.clf()
                    P.close('all')



#------------------------------------------------------------------------------#

#---------------------------------     MAIN     -------------------------------#


# Defining main function
def main():
    
    # Call multi-display function
    maps_multi_display()

            
# Using the special variable
if __name__=="__main__": 
    main()