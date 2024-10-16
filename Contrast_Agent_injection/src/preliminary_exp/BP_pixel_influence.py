# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:34:14 2021

@author: Wistan
"""


import os
import torch
import numpy as np
import pickle
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import pylab as P
import matplotlib.colors as colors
import matplotlib as mpl



#-----------------------------     Parameters     -----------------------------#



# List of trained networks
networks = [
            "resnet",
            # "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Pixel_Influence/'


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Range of values for experiment
range_values = [i for i in range(0, 101)]


# Set up matplot lib figures.
ROWS = 1
COLS = 3
UPSCALE_FACTOR = COLS*1.1999 - 0.2
DPI=72



#-------------------------------     DISPLAY     ------------------------------#



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


# Write & save txt file with stats
def write_stats(stats_max, stats_min, y_pred, pathArch):
    
    # File name
    file = open(pathArch + 'stats_pixels.txt', "a")
    
    # Original output
    file.write('y_pred_original\t' + '%.4f'%y_pred + '\n\n')
    # Columns
    file.write('MAX\t\t' + 'MIN\t\n\n')
    # Original pixel values
    file.write('%.4f'%stats_max[2] + '\t\t' + '%.4f'%stats_min[2] + '\t\t' + 'orig_pixel_value \t\n')
    # Pixel position
    file.write(str(stats_max[1]) + '\t' + str(stats_min[1]) + '\t' + 'pixel_position \t\n')
    # Pixel original attribution
    file.write('%.4f'%stats_max[0] + '\t\t' + '%.4f'%stats_min[0] + '\t\t' + 'original_attr \t\n\n\n')
    
    file.close()


# Create and save curves
def display_curves(df, filepath):
    
    # Create new figure for plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=DPI)
    set_size(12, 12)
    
    # Output evolution plot
    ax1.plot(df['pixel_value'].values.tolist(), df['network_output'].values.tolist())
    # Name the y label
    ax1.set(ylabel='Network Output')
    # Name the graph title
    ax1.set_title('Network Output Evolution')
    
    # Attribution evolution plot
    ax2.plot(df['pixel_value'].values.tolist(), df['BP_attribution'].values.tolist())
    # Name the x and y labels
    ax2.set(xlabel='Value of pixel', ylabel='Saliency Attribution')
    # Name the graph title
    ax2.set_title('Attribution Evolution')
    
    # Add grids
    ax1.grid()
    ax2.grid()
    
    plt.xticks(np.arange(0, 1.01, 0.05))
    # Save the figure
    fig.savefig(filepath)



#------------------------------------------------------------------------------#

#---------------------------------     BP     ---------------------------------#



def BP_im(im_tensor, model):
    
    # Load input image
    im_tensor = im_tensor.to(device)
    
    # Set the requires_grad_ to the image for retrieving gradients
    im_tensor.requires_grad = True

    # Retrieve output from the image
    y_pred = model(im_tensor)[0]

    # Do backpropagation to get the derivative of the output based on the image
    model.zero_grad()
    y_pred.backward()
    grads = im_tensor.grad.data.squeeze()
    attr = grads.cpu().detach().numpy()
    
    y_pred_out = torch.sigmoid(y_pred)[0].detach().cpu().numpy()
    
    return y_pred_out, attr



#------------------------------------------------------------------------------#

#--------------------------     PIXEL INFLUENCE     ---------------------------#



def pixel_influence(im_tensor, model, pathArch):
    
    # BP on original image
    y_pred, attr = BP_im(im_tensor, model)
    
    # To absolute values
    attr_abs = np.abs(attr)
    
    # Original image to numpy
    im_numpy = im_tensor.squeeze().numpy()
    
    
    ### Display image & saliency maps
    
    
    P.figure(figsize=(1, 1), dpi=DPI)
    # Display percentages
    set_size(attr.shape[1]*UPSCALE_FACTOR/DPI, (attr.shape[0]/DPI)+1)

    # Show original image
    ShowImage(im_numpy, title='Input Image', ax=P.subplot(ROWS, COLS, 1))
    # Show Raw heatmap attributions
    ShowHeatMap(attr, title='Saliency Map (original values)', ax=P.subplot(ROWS, COLS, 2))
    # Show Raw heatmap attributions
    ShowHeatMap(attr_abs, title='Saliency Map (absolute values)', ax=P.subplot(ROWS, COLS, 3))
    
    # Save image
    os.makedirs(pathArch, exist_ok=True)
    P.savefig(pathArch + 'Original_image_attr.tiff')
    P.close()
    
    
    ### Find highest & lowest attributions
    
    
    # Find the highest attribution
    attr_max = np.amax(attr)
    attr_max_idx = np.unravel_index(attr.argmax(), attr.shape)
    attr_max_pix_val = im_numpy[attr_max_idx[0], attr_max_idx[1]]
    
    # Find the lowest attribution
    attr_min = np.amin(attr)
    attr_min_idx = np.unravel_index(attr.argmin(), attr.shape)
    attr_min_pix_val = im_numpy[attr_min_idx[0], attr_min_idx[1]]
    
    # Save stats
    stats_max = [attr_max, attr_max_idx, attr_max_pix_val]
    stats_min = [attr_min, attr_min_idx, attr_min_pix_val]
    write_stats(stats_max, stats_min, y_pred, pathArch)
    
    
    ### Experiment on both pixels
    
    
    # Arrays for graph computation
    results_max = np.empty((len(range_values), 3))
    results_min = np.empty((len(range_values), 3))
    
    # Copies of im_tensor
    im_tensor_max = torch.clone(im_tensor)
    im_tensor_min = torch.clone(im_tensor)
    
    # Variation of pixel value
    for idx, i in enumerate(range_values):
        
        # Change value
        im_tensor_max[:, :, attr_max_idx[0], attr_max_idx[1]] = i/100
        im_tensor_min[:, :, attr_min_idx[0], attr_min_idx[1]] = i/100
        
        # Compute attributions again (with outputs)
        y_pred_max, saliency_max = BP_im(im_tensor_max, model)
        y_pred_min, saliency_min = BP_im(im_tensor_min, model)
        
        # Extract specified pixels new attributions
        new_attr_max = saliency_max[attr_max_idx[0], attr_max_idx[1]]
        new_attr_min = saliency_min[attr_min_idx[0], attr_min_idx[1]]
        
        # Save results
        results_max[idx] = (i/100, new_attr_max, y_pred_max)
        results_min[idx] = (i/100, new_attr_min, y_pred_min)
        
    
    # Results arrays to DataFrames
    df_max = pd.DataFrame(results_max, columns=[['pixel_value', 'BP_attribution', 'network_output']])
    df_min = pd.DataFrame(results_min, columns=[['pixel_value', 'BP_attribution', 'network_output']])
    
    # Save as csv
    df_max.to_csv(pathArch + 'results_max.csv')
    df_min.to_csv(pathArch + 'results_min.csv')
    
    # Display curves
    display_curves(df_max, pathArch + 'curves_max.tiff')
    display_curves(df_min, pathArch + 'curves_min.tiff')



#------------------------------------------------------------------------------#

#-------------------------------     MAIN     ---------------------------------#



def main():
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_30.pt')
    
    # One image for this experiment
    one_im = test_images[0].unsqueeze(0)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        # Change results path according to model
        pathArch = pathResults + arch + '/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # Call experiment function
        pixel_influence(one_im, model, pathArch)




# Using the special variable
if __name__=="__main__": 
    main()
