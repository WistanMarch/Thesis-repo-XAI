# -*- coding: utf-8 -*-
"""
| Author: Wistan Marchadour
| Created on: Sept 2022
"""


import os
import numpy as np
from glob import glob
from monai.transforms import LoadImaged, ToTensord, Compose, ToNumpyd
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
# from skimage.measure import compare_psnr
# import cv2
# import nibabel as nib


# Method to test on
method1 = 'IntegratedGradients(Max)'
method2 = 'BPxInput-(Max)'

# Paths
pathRoot = './'
pathInput = pathRoot + 'Test_transformed/'
pathAttr1 = pathRoot + 'Results/Raw_attrs/' + method1 + '/'
pathAttr2 = pathRoot + 'Results/Raw_attrs/' + method2 + '/'


# Load test files names
input_dir = os.listdir(pathInput)

# Load all corresponding maps
transforms = Compose(
    [LoadImaged(keys="image"),
     ToTensord(keys="image"),
     ToNumpyd(keys="image")]
)

# Average similarity scores
mse_avg = 0
psnr_avg = 0
ssim_avg = 0


# For each test file
for filename in input_dir:
    
    input_name = filename[:filename.find('.')]

    # Load all corresponding maps names
    maps_paths_1 = glob(os.path.join(pathAttr1, input_name+'*'), recursive=False)
    maps_paths_2 = glob(os.path.join(pathAttr2, input_name+'*'), recursive=False)
    
    # Get volume from Method 1
    test_data_1 = {'image': maps_paths_1}
    map_1 = transforms(test_data_1)['image']
    map_1 = (map_1 - map_1.min())/(map_1.max() - map_1.min())
    
    # Get volume from Method 2
    test_data_2 = {'image': maps_paths_2}
    map_2 = transforms(test_data_2)['image']
    map_2 = (map_2 - map_2.min())/(map_2.max() - map_2.min())
    
    # Data range of both images
    data_range = np.max([map_1, map_2]) - np.min([map_1, map_2])
    
    # Similarity metrics
    mse = mean_squared_error(map_1, map_2)
    psnr = peak_signal_noise_ratio(map_1, map_2, data_range=1)
    ssim = structural_similarity(map_1, map_2, data_range=1)
    
    # Accumulate similarity metrics over test data
    mse_avg += mse
    psnr_avg += psnr
    ssim_avg += ssim


# Average of all similarity scores
mse_avg /= len(input_dir)
psnr_avg /= len(input_dir)
ssim_avg /= len(input_dir)

# Display results
print("Average Similarity Scores :")
print("\t MSE :", mse_avg)
print("\t PSNR :", psnr_avg)
print("\t SSIM :", ssim_avg)
print("\n")