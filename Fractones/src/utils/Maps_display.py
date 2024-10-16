# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pylab as P
import matplotlib.colors as colors
import matplotlib as mpl


#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathResults = pathRoot + 'Results/'
pathRawAttr = pathResults + 'Raw_attrs/'
pathDisplay = pathResults + 'Methods_Display/'


# All methods for files loading
INTERPRET_METHODS = ['Backpropagation',
                    'Deconvolution',
                    'ExpectedGradients',
                    'IntegratedGradients(Black)',
                    'IntegratedGradients(White)',
                    'IntegratedGradients(BlackWhite)']

# Using original or absolute values
absolute = True

# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72

# Change paths if absolute values
if (absolute):
    pathRawAttr = pathRawAttr[:-1] + ' (absolute)' + pathRawAttr[-1]
    pathDisplay = pathDisplay[:-1] + ' (absolute)' + pathDisplay[-1]


#------------------     Utility methods     ------------------#


# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im[:,:,[2,1,0]], vmin=0, vmax=1)
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

    # Get all subdirectories
    list_all_dirs_files = [x for x in os.walk(pathImages)]
    dirs = list_all_dirs_files[0][1]
    
    # For each subdirectory
    for dirIdx in range(len(dirs)):
        
        dir_path = pathImages + dirs[dirIdx] + '/'
        
        print("Directory :", dir_path)
    
        # Images chosen for application of saliency maps
        images_paths = list_all_dirs_files[dirIdx+1][2]
        
        # Range of number of test slices
        for idx in range (len(images_paths)):
            print('\t Maps Display Slice number', idx+1)
            
            # Load and process input image
            im_numpy = cv2.imread(dir_path + images_paths[idx])
            im_numpy = cv2.resize(im_numpy, (512, 512))
            im_numpy = im_numpy.astype('uint8') / 255
        
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                print("\t\t Running", method)
                
                # Create full attributions path
                pathLoad = pathRawAttr + method + '/' + dirs[dirIdx] + '/'
                pathSave = pathDisplay + method + '/' + dirs[dirIdx] + '/'
                
                # Load raw and XRAI results
                raw_attr = np.load(pathLoad + 'Raw_' +method+ '_' +images_paths[idx]+ '.npy')
                
                P.figure(figsize=(1, 1), dpi=DPI)
                # Set figure size
                set_size(raw_attr.shape[1]*UPSCALE_FACTOR/DPI, (raw_attr.shape[0]/DPI)+1)
            
                # Show original image
                ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
                # Show Raw heatmap attributions
                ShowHeatMap(raw_attr, title='Attribution Map', ax=P.subplot(ROWS, COLS, 2))
                
                # Save image
                P.savefig(pathSave + 'Display_' +method+ '_' +images_paths[idx]+ '.png')
                P.close()
                
# Using the special variable
if __name__=="__main__": 
    main()