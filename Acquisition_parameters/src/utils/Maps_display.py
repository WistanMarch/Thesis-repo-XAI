# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import pickle

from matplotlib import pyplot as plt
from matplotlib import pylab as P
import matplotlib.colors as colors
import matplotlib as mpl


#------------------     Parameters     ------------------#


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/'
pathRawAttr = pathResults + 'Raw_attrs/'
pathDisplay = pathResults + 'Methods_Display/'


# All methods for files loading
INTERPRET_METHODS = [
                    # 'BP',
                    'EG',
                    # 'IGB',
                    # 'IGW',
                    # 'IGBW',
                    ]


# Using original or absolute values
absolute = True


# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_263.pt')
# Load params dictionnary
f = open("./params.pkl", "rb")
params = pickle.load(f)
f.close()


# Set up matplot lib figures.
ROWS = 1
COLS = 2
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72


# Change paths if absolute values
if (absolute):
    pathRawAttr = pathRawAttr[:-1] + '(absolute)' + pathRawAttr[-1]
    pathDisplay = pathDisplay[:-1] + '(absolute)' + pathDisplay[-1]


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
        cmap = 'inferno'
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
    
    # For each parameter prediction
    for param_name, param_values in params.items():
            
        print("Param", param_name)
    
        # Range of methods
        for method in (INTERPRET_METHODS):
            
            print("\t Running", method)
                
            # Create full attributions path
            pathLoad = pathRawAttr + param_name + '/' + method + '/'
            pathSave = pathDisplay + param_name + '/' + method + '/'
            os.makedirs(pathSave, exist_ok=True)
    
            # Get all attributions files names
            maps_names = [f for f in os.listdir(pathLoad) if os.path.isfile(os.path.join(pathLoad, f))]
    
            # For each attributions file
            for filename in maps_names:
                
                # Load raw and XRAI results
                raw_attr = np.load(pathLoad + filename)
                
                # Extract index of slice and label
                filename_no_ext = filename.replace('.npy', '')
                filename_crop = filename_no_ext.replace(method+'_Im_', '')
                sliceIdx = int(filename_crop[0 : filename_crop.find('_')])
                label = filename_crop[filename_crop.rfind('_')+1 : ]
                
                # Convert to numpy
                im_numpy = test_images[sliceIdx][0].detach().numpy()
                
                P.figure(figsize=(1, 1), dpi=DPI)
                # Display percentages
                set_size(raw_attr.shape[1]*UPSCALE_FACTOR/DPI, (raw_attr.shape[0]/DPI)+1)
            
                # Show original image
                ShowImage(im_numpy, title='Input Image ('+label+')', ax=P.subplot(ROWS, COLS, 1))
                # Show Raw heatmap attributions
                ShowHeatMap(raw_attr, title='Attribution Map '+param_name, ax=P.subplot(ROWS, COLS, 2))
                
                # Save image
                P.savefig(pathSave + filename_no_ext+ '.png')
                P.close()
                
# Using the special variable
if __name__=="__main__": 
    main()