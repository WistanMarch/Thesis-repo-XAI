# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib import pylab as P
import matplotlib.colors as colors
import matplotlib as mpl


#------------------     Parameters     ------------------#


# List of trained networks
networks = [
            "resnet",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'


# Methods names for raw & absolute maps
INTERPRET_METHODS = {
                "Raw" : "Backpropagation",
                "Absolute" : "BP_custom",
                    }


# Colormaps to apply for raw & absolute maps
COLORMAPS = {
                "Raw" : ['bwr', 'seismic', 'hsv'],
                "Absolute" : ['viridis', 'inferno', 'gray', 'hot', 'Reds'],
            }


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_30.pt')

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72


#------------------     Utility methods     ------------------#


# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='gray', vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None, colormap='seismic'):
    if ax is None:
        P.figure()
    P.axis('off')
    # Set up the results display
    if (np.min(im) < 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=np.min(im), vcenter=0, vmax=np.max(im))
        cmap = colormap
    elif (np.min(im) >= 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=np.max(im)/2, vmax=np.max(im))
        cmap = colormap
    else:
        print(np.min(im), np.max(im))
        norm = None
        cmap = colormap
    P.imshow(im, cmap=cmap, norm=norm)
    cax = ax.inset_axes([-1, len(im)+5, len(im[0])-1, 4*len(im)/DPI], transform=ax.transData)
    P.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    P.title(title)

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


#------------     Display      ------------#

# Defining main function
def main():
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathRawAttr = pathResults + arch + '/Raw_attrs/'
        pathDisplay = pathResults + arch + '/Methods_Display/'
        
        # For each map configuration
        for map_type in INTERPRET_METHODS:
            
            print("\t Map type " + map_type)
            
            method = INTERPRET_METHODS[map_type]
            
            # Change paths if absolute values
            if (map_type == "Absolute"):
                pathRawAttr = pathRawAttr[:-1] + '(absolute)' + pathRawAttr[-1]
                pathDisplay = pathDisplay[:-1] + '(absolute)' + pathDisplay[-1]
            
            pathLoad = pathRawAttr + method + '/'
            
            # For each colormap
            for colormap in COLORMAPS[map_type]:
                
                print("\t\t Colormap " + colormap)
                
                # Define output folder path
                pathSave = pathDisplay + colormap + '/'
                os.makedirs(pathSave, exist_ok=True)
                
                # Range of number of test slices
                for sliceIdx in range (len(test_images)):
                    
                    im_numpy = test_images[sliceIdx][0].detach().numpy()
                    
                    # Load saliency map
                    raw_attr = np.load(pathLoad + 'Raw_attr_' + method + '_Im_' +str(sliceIdx+1)+ '.npy')
                    
                    P.figure(figsize=(1, 1), dpi=DPI)
                    # Set image size
                    set_size(raw_attr.shape[1]*UPSCALE_FACTOR/DPI, (raw_attr.shape[0]/DPI)+1)
                
                    # Show original image
                    ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
                    # Show saliency map
                    ShowHeatMap(raw_attr, title='Attribution Map '+colormap, ax=P.subplot(ROWS, COLS, 2), colormap=colormap)
                    
                    # Save image
                    P.savefig(pathSave + '/Raw_display_' + method + '_Im_' + str(sliceIdx+1) + '_cmap_' + colormap + '.tiff')
                    P.close()
                
# Using the special variable
if __name__=="__main__": 
    main()