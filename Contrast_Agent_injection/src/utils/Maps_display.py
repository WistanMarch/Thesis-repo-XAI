# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import pylab as P
import matplotlib.colors as colors
import matplotlib as mpl


#------------------     Parameters     ------------------#


# List of trained networks
networks = [
            # "resnet",
            "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathTruthMasks = pathRoot + 'dataset_260/Masks/Full/'


# All methods for files loading
INTERPRET_METHODS = [
                    'BP',
                    # 'Deconv',
                    # 'IG(0)',
                    # 'IG(1)',
                    # 'IG(0-1)',
                    # 'IGA(01)',
                    # 'IGA(10)',
                    # 'EG',
                    # 'GradCAM',
                    # 'ImgReg+BP',
                    # 'ImgReg+IG(0)',
                    # 'ImgReg+EG',
                    # 'ImgReg+GradCAM',
                    # 'SG+BP',
                    # 'SG+Deconv',
                    # 'SG+IGB',
                    # 'SG+IGW',
                    # 'SG+IGBW',
                    # 'SG+IGAW1B0',
                    # 'SG+IGAW0B1',
                    # 'SG+EG',
                    # 'SG+GradCAM',
                    # 'SGsq+BP',
                    # 'SGsq+Deconv',
                    # 'SGsq+IGB',
                    # 'SGsq+IGW',
                    # 'SGsq+IGBW',
                    # 'SGsq+IGAW1B0',
                    # 'SGsq+IGAW0B1',
                    # 'SGsq+EG',
                    # 'SGsq+GradCAM',
                     ]

# Using original or absolute values
absolute = True

# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_260.pt')

# Set up matplot lib figures.
ROWS = 1
COLS = 3
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

def ShowHeatMap(im, title, ax=None, inverseValues=False):
    if ax is None:
        P.figure()
    P.axis('off')
    # Set up the results display
    if (np.min(im) < 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=np.min(im), vcenter=0, vmax=np.max(im))
        cmap = 'seismic'
    elif (np.min(im) >= 0 and np.max(im) > 0):
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=np.max(im)/2, vmax=np.max(im))
        cmap = 'Reds'
    else:
        print(np.min(im), np.max(im))
        norm = None
        cmap = 'seismic'
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

        # Change paths if absolute values
        if (absolute):
            pathRawAttr = pathRawAttr[:-1] + '(absolute)' + pathRawAttr[-1]
            pathDisplay = pathDisplay[:-1] + '(absolute)' + pathDisplay[-1]
    
        # Range of number of test slices
        for sliceIdx in range (len(test_images)):
            print('\t Maps Display Slice number', sliceIdx+1)
            
            im_numpy = test_images[sliceIdx][0].detach().numpy()
            
            # Load ground truth mask
            truthMask = np.asarray(Image.open(pathTruthMasks + 'mask_' +str(sliceIdx+1)+ '.tiff').convert('L'))
        
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                print("\t\t Running", method)
                
                # Create full attributions path
                pathLoad = pathRawAttr + method + '/'
                
                # Load raw and XRAI results
                raw_attr = np.load(pathLoad + 'Raw_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy')
                
                P.figure(figsize=(1, 1), dpi=DPI)
                # Display percentages
                set_size(raw_attr.shape[1]*UPSCALE_FACTOR/DPI, (raw_attr.shape[0]/DPI)+1)
            
                # Show original image
                ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
                # Show ground truth
                ShowImage(truthMask, title='Reference Mask', ax=P.subplot(ROWS, COLS, 2))
                # Show Raw heatmap attributions
                ShowHeatMap(raw_attr, title='Attribution Map', ax=P.subplot(ROWS, COLS, 3))
                
                # Save image
                os.makedirs(pathDisplay + method + '/', exist_ok=True)
                P.savefig(pathDisplay + method + '/Raw_display_' +method+ '_Im_' + str(sliceIdx+1) + '.tiff')
                P.close()
                
# Using the special variable
if __name__=="__main__": 
    main()