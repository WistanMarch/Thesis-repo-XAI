# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:30:47 2021

@author: Wistan
"""


import os
import numpy as np
import torch
import saliency.core as saliency


#------------------     Parameters     ------------------#


# List of trained networks
networks = [
            "resnet",
            # "vgg19",
            "Xception",
            ]


# Base paths
pathRoot = './'

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
                    # 'Random',
                     ]


# Apply XRAI on absolute values or not
abs_values = [
                False,
                True
             ]


#------------------     XRAI Full     ------------------#


# Defining main function
def main(): 

    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_260.pt')
    
    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/'
        pathRawAttr = pathResults + 'Raw_attrs/'
        pathDisplay = pathResults + 'XRAI_attrs/'
        
        # For vanilla / absolute maps
        for absolute in abs_values:
            print("\t Absolute : " + str(absolute))

            # Change paths if absolute values
            if (absolute):
                pathRawAttr = pathRawAttr[:-1] + '(absolute)' + pathRawAttr[-1]
                pathDisplay = pathDisplay[:-1] + '(absolute)' + pathDisplay[-1]
        
            for method in INTERPRET_METHODS:
            
                print("\t\t Start " +method+ " + XRAI...")
                    
                for sliceIdx in range(len(test_images)):
                    
                    print('\t\t\t Slice number', sliceIdx+1)
                    
                    # Load input image and raw attributions for given slice and method
                    im_tensor = test_images[sliceIdx]
                    im_numpy = im_tensor.squeeze().numpy() * 255.0
                    raw_attr = np.squeeze(np.load(pathRawAttr + method + '/Raw_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy'))
                    
                    # Construct the saliency object. This alone doesn't do anything.
                    xrai_object = saliency.XRAI()
                    
                    # Launch XRAI with raw attributions (indicate it is attribution map and not input image)
                    xrai_attributions = xrai_object.GetMask(x_value=im_numpy, call_model_function=None, base_attribution=raw_attr)
                    
                    # Save XRAI attributions as npy file
                    os.makedirs(pathDisplay + method + '/', exist_ok=True)
                    np.save(pathDisplay + method + '/XRAI_attr_' +method+ '_Im_' +str(sliceIdx+1)+ '.npy', xrai_attributions)
                    
                print("\t\t End " +method+ " + XRAI...")
                
        
# Using the special variable
if __name__=="__main__": 
    main()