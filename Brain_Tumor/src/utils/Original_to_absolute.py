# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:45:07 2022

@author: Wistan
"""


import numpy as np
import os
from glob import glob
import nibabel as nib
from monai.transforms import LoadImaged, ToTensord, Compose


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathOriginalAttr = pathResults + 'Raw_attrs/'
pathAbsoluteAttr = pathResults + 'Raw_attrs(absolute)/'


# All methods for files loading
INTERPRET_METHODS = [
                    'IG(min)',
                    'IG(0)',
                    'IG(max)',
                    'IG(min-0)',
                    'IG(0-max)',
                    'IG(min-max)',
                    'IG(min-0-max)',
                    ]


# Defining main function
def main(): 

    # Range of methods
    for method in (INTERPRET_METHODS):
        print('Conversion to Absolute', method)
        
        # Create full attributions path
        pathLoad = pathOriginalAttr + method + '/'
        
        # Find all files names
        maps_paths = glob(os.path.join(pathLoad, '*'), recursive=True)
        
        for step, file_path in enumerate(maps_paths):
            
            # Load all corresponding maps
            transform = Compose([
                            LoadImaged(keys='image'),
                            ToTensord(keys='image')
                        ])
            
            dict_data = {'image': file_path}
            result = transform(dict_data)
            data_numpy = result['image'].numpy()
            
            # Convert to absolute
            data_abs = np.abs(data_numpy)
            
            # Save std as .nii file
            pathSave = pathAbsoluteAttr + method + '/'
            os.makedirs(pathSave, exist_ok=True)
            ni_abs = nib.Nifti1Image(data_abs, affine=result['image_meta_dict']['affine'])
            filename = file_path[file_path.rfind('\\')+1:]
            nib.save(ni_abs, pathSave + filename)
        
        
# Using the special variable
if __name__=="__main__": 
    main()