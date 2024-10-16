# -*- coding: utf-8 -*-
"""
| Author: Wistan Marchadour
| Created on: Sept 2022
"""


import os
import numpy as np
from glob import glob
from monai.transforms import LoadImaged, ToTensord, Compose
import nibabel as nib


method = 'Backpropagation'


input_path = './GliMetX_LATIM/Test/'
maps_path = './Results/Raw_attrs/' + method + '/'
std_path = './Results/Prediction_Reproducibility/'


# Load test files names
input_dir = os.listdir(input_path)

# For each test file
for input_name in input_dir:
    
    print(input_name)

    # Load all corresponding maps names
    maps_files = glob(os.path.join(maps_path, input_name+'*'), recursive=False)

    # Load all corresponding maps
    transform = Compose([
                    LoadImaged(keys='image'),
                    ToTensord(keys='image')
                ])
    
    test_data = {'image': maps_files}
    result = transform(test_data)
    loaded_data = result['image']
    
    # Identify the average map
    avg_map = loaded_data[0].numpy()
    model_maps = loaded_data[1:].numpy()
    
    # Perform std on remaining maps
    std_map = np.std(model_maps, axis=0)
    
    # Save std as .nii file
    ni_std = nib.Nifti1Image(std_map, affine=result['image_meta_dict']['affine'])
    nib.save(ni_std, std_path + method + '_' + input_name + '_STD.nii')
