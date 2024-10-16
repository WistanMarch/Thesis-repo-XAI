# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:01:07 2021

@author: Wistan
"""



import os
import math
import torch
import pickle
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

import resnet, xception



#-----------------------------     PARAMETERS     -----------------------------#



# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Intermediate_features/'


# Display parameters
DPI = 72
bar_decal = 150
bar_dims = [0.05, 0.05, 0.9, 0.03]


# Image for intermediate features
im_idx = 6


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


activation = {}



#------------------------------------------------------------------------------#

#-------------------------------     UTILS     --------------------------------#



def forward_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



def backward_activation(name):
    def hook(mod, grad_input, grad_output):
        activation[name] = grad_input[0].clone()
    return hook



def display_save(act, pathSave, module):
    
    # Convert to numpy
    act_np = act.numpy()
    
    # Get extreme values
    if (np.amin(act_np) == 0):
        vmax = np.amax(act_np)
        vmin = np.amin(act_np)
        cmap = 'Reds'
    else:
        vmax = max(np.amax(act_np), abs(np.amin(act_np)))
        vmin = -1 * vmax
        cmap = 'seismic'
    
    # Setup figure
    space_pix = int(0.15 * act.size(-1))
    square_dim = (int)(math.ceil(math.sqrt(act.size(0))))
    fig_width = (act.size(2) * square_dim + space_pix * (square_dim+1)) / DPI
    fig_height = (act.size(1) * square_dim + space_pix * (square_dim+1) + bar_decal) / DPI
    cursor_x = space_pix
    cursor_y = space_pix + bar_decal
    
    # Init figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbaxes = fig.add_axes(bar_dims)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cbaxes, orientation='horizontal')
    
    # Plot images
    for idx in range(act.size(0)):
        fig.figimage(act[idx].squeeze(), xo=cursor_x, yo=cursor_y, origin='upper', cmap=cmap)
        
        if (idx%square_dim == (square_dim-1)):
            cursor_x = space_pix
            cursor_y += act.size(1) + space_pix
        else:
            cursor_x += act.size(2) + space_pix
    
    os.makedirs(pathSave, exist_ok=True)
    fig.savefig(pathSave + 'Im_' + str(im_idx) + '_Layer_' + str(module[0]) + '_' + module[1][0] + '.tiff', facecolor=fig.get_facecolor(), edgecolor='none', dpi=DPI)
    fig.clf()



#------------------------------------------------------------------------------#

#------------------------     INTERMEDIATE MAPS     ---------------------------#



def forward_intermediate_maps(module, model, im):
    
    ### FORWARD HOOK
    forward_hook = module[1][1].register_forward_hook(forward_activation(module[1]))
    output = model(im)
    act = activation[module[1]].squeeze()
    
    ### REMOVE FORWARD HANDLE
    forward_hook.remove() ## hook delete
    activation.clear()
    
    return act.cpu().detach()
    


def BP_intermediate_maps(module, model, im):
    
    im.requires_grad = True
    
    ### BACKWARD HOOK
    backward_hook = module[1][1].register_full_backward_hook(backward_activation(module[1][0]))
    output = model(im)
    output.retain_grad()
    output.backward()
    act = activation[module[1][0]].squeeze()
    
    ### REMOVE BACKWARD HANDLE
    backward_hook.remove() ## hook delete
    im.requires_grad = False
    activation.clear()
    
    return act.cpu().detach()



#------------------------------------------------------------------------------#

#------------------------------     MAIN     ----------------------------------#



def main():
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_30.pt').to(device)
    im = test_images[im_idx].unsqueeze(0)
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathArch = pathResults + arch + '/'
    
        # Load model
        model = pickle.load(open('./serialized_' + arch + '.sav', 'rb')).to(device)
        
        # For each network module
        for module in enumerate(model.named_modules()):
            
            # Take main layers of both networks
            if (arch == "resnet" and isinstance(module[1][1], resnet.Bottleneck)) or (arch == "Xception" and isinstance(module[1][1], xception.Block)):
                
                print("\t Forward")
                
                act_forward = forward_intermediate_maps(module, model, im)
                display_save(act_forward, pathArch+'Forward/', module)
                
                print("\t Backward")
                
                act_backward = BP_intermediate_maps(module, model, im)
                display_save(act_backward, pathArch+'Backward/', module)




# Using the special variable
if __name__=="__main__": 
    main()
