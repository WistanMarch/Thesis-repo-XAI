# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:28:24 2021

@author: Wistan
"""


import os
import torch
import numpy as np
from matplotlib import pylab as P

from skimage.segmentation import slic, felzenszwalb, quickshift
from skimage.segmentation import mark_boundaries

from utils_display import set_size, ShowImage, ShowHeatMap



#-----------------------------     Parameters     -----------------------------#


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# All methods for files loading
INTERPRET_METHODS = [
                        'BP',
                        # 'IG(0)',
                    ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
             }


# Tested number of segments
TECHNIQUES = [
              'SLIC',
              'Felzenswalb',
              'Quickshift',
             ]


# Tested number of segments
NUMBER_SEGMENTS = [
                    50,
                    100,
                    200,
                    500,
                    # 600,
                  ]


# Set up matplotlib figures
ROWS = 1
COLS = 3 + len(TECHNIQUES) * len(NUMBER_SEGMENTS) * 2
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72



#------------------------------------------------------------------------------#

#--------------------------------     UTILS     -------------------------------#



# Average saliency map over the segments
def fill_segmentation(values, segmentation, nb_segments):
    
    nb_pix_zone = np.zeros(nb_segments+1)
    sum_zone = np.zeros(nb_segments+1)
    
    for i in range (values.shape[0]):
        for j in range (values.shape[1]):
            idx_seg = segmentation[i][j]
            nb_pix_zone[idx_seg] += 1
            sum_zone[idx_seg] += values[i][j]
    
    avg_zone = sum_zone / nb_pix_zone
    out = np.zeros(segmentation.shape)
    
    for i in range(len(avg_zone)):
        out[segmentation == i] = avg_zone[i]
    
    return out



#------------------------------------------------------------------------------#

#--------------------------------     MAIN     --------------------------------#



# Defining main function
def main():
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt').squeeze()
    
    # For each original image
    for idx, img_tensor in enumerate(test_images):
        
        print("Image n°" + str(idx+1))
        
        # Convert to numpy
        img_np = img_tensor.numpy()
        
        # For each network
        for arch in networks:
            
            # print("\t Network " + arch)
            
            # For each map type
            for map_type in MAP_TYPES:
                
                # print("\t\t" + map_type)
                
                # For each method
                for method in INTERPRET_METHODS:
                    
                    # print("\t\t\t" + method)
                    
                    # Save path for displays
                    pathSave = pathResults + 'Classic_Segmentation/' + arch + '/' + map_type + '/' + method + '/'
                    os.makedirs(pathSave, exist_ok=True)
                    
                    # Path of the saliency maps
                    pathMap = pathResults + arch + '/' + MAP_TYPES[map_type] + '/' + method + '/'
                    attr = np.load(pathMap + 'Raw_attr_' + method + '_Im_' + str(idx+1) + '.npy')
                    
                    # Init figure
                    P.figure(figsize=(1, 1), dpi=DPI)
                    # Display percentages
                    set_size(img_np.shape[1]*UPSCALE_FACTOR/DPI, (img_np.shape[0]/DPI)+1)
                    
                    # Show original image
                    ShowImage(img_np, title='Input', ax=P.subplot(ROWS, COLS, 1))
                    # Show Raw heatmap attributions
                    ShowHeatMap(attr, title='Absolute Saliency Map', ax=P.subplot(ROWS, COLS, 2))
                    
                    # For each tested number of segments
                    for idx_nb, nb_seg in enumerate(NUMBER_SEGMENTS):
                        
                        # For each segmentation technique
                        for idx_tech, technique in enumerate(TECHNIQUES):
                            
                            # Apply each technique
                            if (technique == 'SLIC'): segments = slic(img_np, n_segments=nb_seg, compactness=0.3, max_num_iter=100, sigma=0.6, start_label=0, channel_axis=None, enforce_connectivity=False)
                            elif (technique == 'Felzenswalb'): segments = felzenszwalb(img_np, scale=nb_seg, sigma=0.2, min_size=10)
                            elif (technique == 'Quickshift'): segments = quickshift(np.stack((img_np,)*3, axis=-1), kernel_size=3, max_dist=1000/nb_seg, ratio=0.5)
                            
                            # Get number of segments
                            nb_segments = np.amax(segments)
                            # Shows segments limits on image
                            img_boundaries = mark_boundaries(img_np, segments, color=(1, 0, 0))
                            
                            # Compute average and fills test image with it
                            attr_seg = fill_segmentation(attr, segments, nb_segments)
                            
                            # Show original + boundaries
                            ShowImage(img_boundaries, title='Segmented Input '+technique+' '+str(nb_seg), ax=P.subplot(ROWS, COLS, 3+2*(idx_nb*len(TECHNIQUES)+idx_tech)))
                            # Show Raw heatmap attributions
                            ShowHeatMap(attr_seg, title='Segmented Saliency Map'+technique+' '+str(nb_seg), ax=P.subplot(ROWS, COLS, 4+2*(idx_nb*len(TECHNIQUES)+idx_tech)))
                    
                    # Save image
                    P.savefig(pathSave + 'Im_' + str(idx+1) + '.tiff')
                    P.close()


# Using the special variable
if __name__=="__main__": 
    main()