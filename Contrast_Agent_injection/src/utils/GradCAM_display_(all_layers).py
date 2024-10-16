# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl



#------------------------------------     Parameters     ------------------------------------#


# List of trained networks
networks = [
            "resnet",
            "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/GradCAM/'


# Set up matplot lib figures
DPI=72


#------------------------------     Loading     -------------------------------#


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_290.pt')
# Corresponding labels
test_labels = torch.load(pathRoot + 'test_labels_290.pt').numpy()


#--------------------------     Utility methods     ---------------------------#


def ShowHeatMap(im, title, ax=None):
    if ax is None:
        plt.figure()
    ax.axis('off')
    
    # Set up the results display
    if (np.min(im) < 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=np.min(im), vcenter=0, vmax=np.max(im))
    elif (np.min(im) >= 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=(np.min(im)+np.max(im))/2, vmax=np.max(im))
    else:
        norm = None
    
    cmap = 'viridis'
    ax.imshow(im, cmap=cmap, norm=norm)
    cax = ax.inset_axes([-1, len(im)+5, len(im[0])-1, 4*len(im)/DPI], transform=ax.transData)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    ax.set_title(title)
    
    

# Change size of Matplotlib figure to have definite size of graph (or image)
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


#--------------------------------     MAIN      -------------------------------#


# Defining main function
def main():
    
    # For each network
    for arch in networks:
        
        print("Network " + arch)
        
        # # Load & Save paths
        pathRawAttr = pathResults + arch + '/Raw_attrs/'
        pathDisplay = pathResults + arch + '/Raw_display/'
        os.makedirs(pathDisplay, exist_ok=True)
        
        nb_last_empty = 0

        # Range of number of test slices
        for sliceIdx in range (len(test_images)):
            print('\t GradCAM display slice', sliceIdx+1)
            
            # Load input image
            im_numpy = test_images[sliceIdx][0].detach().numpy()
            # im_color = cv2.cvtColor(im_numpy, cv2.COLOR_GRAY2RGB)
            
            # Load maps filenames and sort along layer number
            filenames = [path for path in os.listdir(pathRawAttr) if ('Slice_'+str(sliceIdx)+'_' in path)]
            layers_nb = [int(name[name.rfind('_')+1:name.rfind('.')]) for name in filenames]
            filenames = [x for _, x in sorted(zip(layers_nb, filenames))]
            
            # Set up matplot lib figures
            ROWS = int(np.ceil(np.sqrt(len(layers_nb))))
            COLS = int(np.round(np.sqrt(len(layers_nb))))
            UPSCALE_FACTOR = COLS*1.1999 - 0.2
            
            # Create figure
            # figInfused, axInfused = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(1, 1), dpi=DPI)
            # set_size(im_numpy.shape[1]*UPSCALE_FACTOR/DPI, (im_numpy.shape[0]*UPSCALE_FACTOR/DPI)+1)
            figGradCAM, axGradCAM = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(1, 1), dpi=DPI)
            set_size(im_numpy.shape[1]*UPSCALE_FACTOR/DPI, (im_numpy.shape[0]*UPSCALE_FACTOR/DPI)+1)
            
            # For each layer map
            for idx, file in enumerate(filenames):
                
                # Load map
                raw_attr = np.load(pathRawAttr + file)
                
                # Count empty last layer maps
                if (idx == len(filenames)-1 and np.min(raw_attr) == 0 and np.max(raw_attr) == 0):
                    nb_last_empty += 1
                
                # Convert to colormap
                # cm = plt.get_cmap('viridis')
                # gcam = cm(raw_attr)
                
                # Find layer name
                layer_name = file[file.rfind('Layer'):file.rfind('.')]
                # # Combine layer map with input image
                # output = im_color * 0.5 + gcam * 0.5
                # Show Raw heatmap attributions
                # ShowHeatMap(output, title=layer_name, ax=axInfused[idx//COLS][idx%COLS])
                ShowHeatMap(raw_attr, title=layer_name, ax=axGradCAM[idx//COLS][idx%COLS])
                # ShowHeatMap(gcam, title=layer_name, ax=axGradCAM[idx//COLS][idx%COLS])
        
            # Save image
            # figInfused.savefig(pathDisplay + 'Display_Slice_' + str(sliceIdx) + '_Infused.tiff')
            figGradCAM.savefig(pathDisplay + 'Display_Slice_' + str(sliceIdx) + '_GradCAM.tiff')
            # plt.close(figInfused)
            plt.close(figGradCAM)
            
        print('Found', nb_last_empty, 'empty last layer maps')


#-----------------------------     CALL MAIN      -----------------------------#


# Using the special variable
if __name__=="__main__":
    main()