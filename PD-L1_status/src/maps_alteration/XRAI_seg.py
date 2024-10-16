# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:30:47 2021

@author: Wistan
"""


"""
Using XRAI segmentation algorithm on 3D attribution maps
Since XRAI does not support 3D, splitting 3D maps into 2D maps along vertical axis,
computing XRAI on each, and recombining
"""


import os
import numpy as np
import torch
import nibabel as nib
import saliency.core as saliency

from utils.XAI_utils import XAI_dataset


#------------------     Parameters     ------------------#


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


# All methods to apply
INTERPRET_METHODS = [
                        # 'BP',
                        # 'Deconv',
                        # 'GradCAM',
                        'IG(Zero)',
                        'IG(Min)',
                        'IG(Max)',
                        'IG(Zero-Min)',
                        'IG(Zero-Max)',
                        'IG(Min-Max)',
                        'IG(Avg)',
                        # 'EG',
                     ]


# Absolute values (no/yes/both)
abs_values = {
                False : '',
                True : '(absolute)',
             }


# Percentages of most important regions
importancePerc = [i for i in range(0, 101)]


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


#------------------     XRAI Full     ------------------#


# Defining main function
def main(): 

    # Output path
    pathResults = loading + 'Results/'
    
    # For each trial (=fold)
    for trial in trials:
        
        print("Trial", trial)
        
        # Trial folder name
        trial_folder = 'immugast-3D-'+params['feature']+'-c-'+str(params['cutoff'])+'-t-'+str(trial)+'-n-'+str(params['net_id'])+'-b-'+str(params['batch'])+'-s-'+str(params['size'])+'/'
        
        # Create dataset for XAI
        ids, _, loader, _ = XAI_dataset(params, trial, device, loading)
        
        # For each element in loader
        for step, data in enumerate(loader):
            
            # Display id
            print("\t ID", ids[step])
            
            # Load original input & affine matrix
            in_tensor = data[0].to(device)
            in_numpy = in_tensor.squeeze().cpu().numpy()
            affine = data[0].meta['affine'].squeeze()
            
            # List of all segments
            segments = []
            
            # For each slice along VERTICAL axis
            for idx in range(in_numpy.shape[-1]):
                
                # Extract 2D slice
                slice2D = in_numpy[..., idx]
                
                # Call XRAI segmentation function to get segments
                segs_slice = saliency.xrai._get_segments_felzenszwalb(slice2D, resize_image=False)
                # Append to overall list
                segments.append(segs_slice)
                
            # For both original & absolute maps
            for absolute in abs_values:
                
                print("\t\t Absolute : " + str(absolute))
                
                # Change paths depending on absolute values or not
                pathRawAttr = pathResults + 'Raw_attrs' + abs_values[absolute] + '/'
                pathXRAIAttr = pathResults + 'XRAI_attrs' + abs_values[absolute] + '/'
                
                # For each method
                for method in INTERPRET_METHODS:
                
                    print("\t\t\t XRAI " + method)
                    
                    # Load attribution map
                    attrPath = pathRawAttr + method + '/' + trial_folder
                    attrName = 'Raw_attrs' + abs_values[absolute] + '_' + method + '_' + ids[step] + '.nii'
                    ni_map_attr = nib.load(attrPath + attrName)
                    map_attr = ni_map_attr.get_fdata()
                    
                    # Reconstructed 3D XRAI attribution map
                    xrai_attributions = np.zeros_like(map_attr)
                    
                    # For each slice along VERTICAL axis
                    for idx in range(map_attr.shape[-1]):
                        
                        # Extract 2D slice of input & map
                        slice2D = in_numpy[..., idx]
                        map_slice2D = map_attr[..., idx]
                        # Segments for specified slice
                        segs = segments[idx]
                        
                        # Construct the saliency object. This alone doesn't do anything.
                        xrai_object = saliency.XRAI()
                        
                        # Launch XRAI with input slice / map slice / segments
                        xrai_attr_slice = xrai_object.GetMask(x_value=slice2D, call_model_function=None, segments=segs, base_attribution=map_slice2D)
                        
                        # Reconstruct 3D attribution map
                        xrai_attributions[..., idx] = xrai_attr_slice
                
                    # Save XRAI attributions as .nii
                    savePath = pathXRAIAttr + method + '/' + trial_folder
                    saveName = 'XRAI_attrs' + abs_values[absolute] + '_' + method + '_' + ids[step] + '.nii'
                    os.makedirs(savePath, exist_ok=True)
                    ni_img = nib.Nifti1Image(xrai_attributions, affine=affine)
                    nib.save(ni_img, savePath + saveName)
                
                
        
# Using the special variable
if __name__=="__main__": 
    main()