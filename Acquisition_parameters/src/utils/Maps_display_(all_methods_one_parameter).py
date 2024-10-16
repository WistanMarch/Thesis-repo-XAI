# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:34:15 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import pickle
from PIL import Image
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
INTERPRET_METHODS = ['BP',
                     'EG',
                     'IGB',
                     'IGW',
                     'IGBW']

# Using original or absolute values
absolute = True

# Images chosen for application of saliency maps
test_images = torch.load(pathRoot + 'test_slices_263.pt')
# Load params dictionnary
f = open("./params.pkl", "rb")
params = pickle.load(f)
f.close()

# To be removed if all params are wanted
del params['kilo_voltage']

# Set up matplot lib figures.
ROWS = 1
COLS = 1 + len(INTERPRET_METHODS)
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
    
    # For each parameter prediction
    for param_name, param_values in params.items():
            
        print("Param", param_name)
    
        # Create full attributions path
        pathLoad = pathRawAttr + param_name + '/'
        
        # Get paths of all files in dir and sub-dirs
        all_files = list()
        for (dirpath, dirnames, filenames) in os.walk(pathLoad):
            all_files += [os.path.join(dirpath, file) for file in filenames]
        
        # For each slice
        for sliceIdx in range (len(test_images)):
                
            print("\t Slice number", sliceIdx+1)
            
            # Get all files corresponding to the index
            filenames_matches = [x for x in all_files if x.find('_'+str(sliceIdx+1)+'_') != -1]
            
            # Convert to numpy
            im_numpy = test_images[sliceIdx][0].detach().numpy()
            
            # Create figure
            P.figure(figsize=(1, 1), dpi=DPI)
            figIdx = 2
            # Resize figure
            set_size(im_numpy.shape[1]*UPSCALE_FACTOR/DPI, (im_numpy.shape[0]/DPI)+1)
        
            # Show original image
            ShowImage(im_numpy, title='Input Image ('+param_name+')', ax=P.subplot(ROWS, COLS, 1))
            
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                # Find corresponding path
                attr_path = [x for x in filenames_matches if x.find(method) != -1][0]
                
                # Load raw and XRAI results
                raw_attr = np.load(attr_path)
                
                # Show Raw heatmap attributions
                ShowHeatMap(raw_attr, title='Attribution Map '+method, ax=P.subplot(ROWS, COLS, figIdx))
                figIdx += 1
                
            # Extract part of files names to re-use
            filename_only = filenames_matches[0][filenames_matches[0].find('\\') :-1]
            filename_end = filename_only[filename_only.find('_')+1 : filename_only.rfind('.')]
                
            # Save image
            P.savefig(pathDisplay + param_name + '/combined/' + param_name + '_' + filename_end + '.png')
            P.close()
                
                
# Using the special variable
if __name__=="__main__": 
    main()