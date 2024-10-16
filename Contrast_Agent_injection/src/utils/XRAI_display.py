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


# Base paths
pathRoot = './'
pathTruthMasks = pathRoot + 'dataset_260/Masks/Full/'


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# All methods for files loading
INTERPRET_METHODS = [
                        # 'BP',
                        # 'Deconv',
                        'IG(0)',
                        'IG(1)',
                        'IG(0-1)',
                        'IGA(01)',
                        'IGA(10)',
                        # 'EG',
                        # 'GradCAM',
                        # 'ImgReg_BP',
                        # 'ImgReg_IntegratedGradients(0)',
                        # 'ImgReg_EG',
                        # 'ImgReg_GradCAM',
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


# Percentages of most important regions
importancePerc = [3, 5, 10, 25]


# Using original or absolute values
absolute = True


# Set up matplot lib figures.
ROWS = 1
COLS = 4 + len(importancePerc)
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
    elif (np.min(im) < 0 and np.max(im) <= 0):
        print(np.min(im), np.max(im))
        norm = colors.TwoSlopeNorm(vmin=np.min(im), vcenter=np.min(im)/2, vmax=0)
        cmap = (plt.cm.get_cmap('Blues')).reversed()
    else:
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

    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt')
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/'
        pathRawAttr = pathResults + 'Raw_attrs/'
        pathXRAIAttr = pathResults + 'XRAI_attrs/'
        pathDisplay = pathResults + 'XRAI_Display/'

        # Change paths if absolute values
        if (absolute):
            pathRawAttr = pathRawAttr[:-1] + '(absolute)' + pathRawAttr[-1]
            pathXRAIAttr = pathXRAIAttr[:-1] + '(absolute)' + pathXRAIAttr[-1]
            pathDisplay = pathDisplay[:-1] + '(absolute)' + pathDisplay[-1]
    
        # Range of number of test slices
        for sliceIdx in range (len(test_images)):
            
            print("\t XRAI Display Slice number", sliceIdx+1)
            
            im_numpy = test_images[sliceIdx][0].detach().numpy()
            
            # Load ground truth mask
            truthMask = np.asarray(Image.open(pathTruthMasks + 'mask_' +str(sliceIdx+1)+ '.tiff').convert('L'))
        
            # Range of methods
            for method in (INTERPRET_METHODS):
                
                print("\t\t Running", method)
                
                # Load raw and XRAI results
                raw_attr = np.load(pathRawAttr + method + '/Raw_attr_' + method + '_Im_' +str(sliceIdx+1)+ '.npy')
                xrai_attr = np.load(pathXRAIAttr + method + '/XRAI_attr_' + method + '_Im_' +str(sliceIdx+1)+ '.npy')
                
                P.figure(figsize=(1, 1), dpi=DPI)
                # Display percentages
                set_size(xrai_attr.shape[1]*UPSCALE_FACTOR/DPI, (xrai_attr.shape[0]/DPI)+1)
            
                # Show original image
                ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
                # Show ground truth
                ShowImage(truthMask, title='Reference Mask', ax=P.subplot(ROWS, COLS, 2))
                # Show Raw heatmap attributions
                ShowHeatMap(raw_attr, title='Pixel-wise map', ax=P.subplot(ROWS, COLS, 3))
                # Show XRAI heatmap attributions
                ShowHeatMap(xrai_attr, title='Region-wise map', ax=P.subplot(ROWS, COLS, 4))
                
                for i in range(len(importancePerc)):
                    # Show most salient % of the image
                    mask = xrai_attr > np.percentile(xrai_attr, 100-importancePerc[i])
                    im_mask = np.array(im_numpy)
                    im_mask[~mask] = 0
                    ShowImage(im_mask, title='XRAI '+str(importancePerc[i])+'%', ax=P.subplot(ROWS, COLS, 5+i))
                    
                # Save image
                os.makedirs(pathDisplay + method + '/', exist_ok=True)
                P.savefig(pathDisplay + method + '/XRAI_display_' +method+ '_Im_' + str(sliceIdx+1) + '.png')
                P.close()
                
            
# Using the special variable
if __name__=="__main__": 
    main()