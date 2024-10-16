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
            "resnet",
            "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathTruthMasks = pathRoot + 'dataset_260/Masks/Full/'


# All methods for files loading
INTERPRET_METHODS = [
                        'BP',
                        # 'SG+BP',
                        # 'SGSQ+BP',
                        # 'ImgReg+BP',
                        'IG(0)',
                        # 'SG+IG(0)',
                        # 'SGSQ+IG(0)',
                        # 'ImgReg+IG(0)',
                        'EG',
                        # 'SG+EG',
                        # 'SGSQ+EG',
                        # 'ImgReg+EG',
                        'GradCAM',
                        # 'SG+GradCAM',
                        # 'SGSQ+GradCAM',
                        # 'ImgReg+GradCAM',
                        'Avg(BP.IG(0).GradCAM.EG)',
                        'Product(BP.IG(0).GradCAM.EG)',
                        # 'Avg(IG(0).GradCAM)',
                        # 'Product(IG(0).GradCAM)',
                     ]


# All Map Types to load
MAP_TYPES = {
            'pix_abs' : 'Raw_attrs(absolute)',
            # 'pix_orig' : 'Raw_attrs',
             }


# Exclude mispredicted slices (idx starts at 0)
excluded_idx = [
                # 262,        # For 290 images workset
                232,        # For 260 images workset
               ]


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_260.pt')


# Set up matplot lib figures.
ROWS = 1
COLS = 1 + len(INTERPRET_METHODS)
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
        
        # For each map type
        for map_type in MAP_TYPES:
            
            pathType = pathResults + arch + '/' + MAP_TYPES[map_type] + '/'
            pathDisplay = pathResults + arch + '/Methods_Display_multiple_methods/'
            
            # Range of number of test slices
            for sliceIdx in range (len(test_images)):
                
                if (sliceIdx not in excluded_idx):
                    
                    print('\t Input n°' + str(sliceIdx+1))
                    
                    im_numpy = test_images[sliceIdx][0].detach().numpy()
                    
                    P.figure(figsize=(1, 1), dpi=DPI)
                    # Display percentages
                    set_size(im_numpy.shape[1]*UPSCALE_FACTOR/DPI, (im_numpy.shape[0]/DPI)+1)
                    
                    # Show original image
                    ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
                    
                    # Range of methods
                    for m_idx, method in enumerate(INTERPRET_METHODS):
                        
                        # print("\t\t Running", method)
                        
                        # Create full attributions path
                        pathMethod = pathType + method + '/'
                        
                        # Load raw and XRAI results
                        raw_attr = np.load(pathMethod + 'Raw_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy')
                        
                        # Show original saliency map
                        ShowHeatMap(raw_attr, title=method + ' Saliency Map', ax=P.subplot(ROWS, COLS, 2+m_idx))
                    
                    # Save image
                    os.makedirs(pathDisplay + str(INTERPRET_METHODS) + '/', exist_ok=True)
                    P.savefig(pathDisplay + str(INTERPRET_METHODS) + '/Display_Im_' + str(sliceIdx+1) + '_' + map_type + '.tiff')
                    P.close()
                
# Using the special variable
if __name__=="__main__": 
    main()