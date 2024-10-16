# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:45:07 2022

@author: Wistan
"""


import os
import numpy as np


# List of trained networks
networks = [
            "resnet",
            "Xception",
            ]


# Base paths
pathRoot = './'


# All methods for files loading
INTERPRET_METHODS = [
                    'BP',
                    # 'Deconv',
                    'IG(0)',
                    # 'IG(1)',
                    # 'IG(0-1)',
                    # 'IGAW1B0',
                    # 'IGAW0B1',
                    'EG',
                    'GradCAM',
                     ]



# Defining main function
def main():

    # For each network
    for arch in networks:
            
        print("Network " + arch)
        
        pathResults = pathRoot + 'Results/' + arch + '/SG_SGSQ_Optim/'
        pathOriginalAttr = pathResults + 'Raw_attrs/'
        pathAbsoluteAttr = pathResults + 'Raw_attrs(absolute)/'
        
        # Range of methods
        for method in (INTERPRET_METHODS):
            print('\t Conversion to Absolute Method', method)
            
            # Create full attributions path
            pathLoad = pathOriginalAttr + method + '/'
            
            # Get all npy files names
            pathsAttrib = [file for file in os.listdir(pathLoad) if (file.startswith('Raw_attr_SG'))]
    
            # Range of maps paths
            for pathSG in pathsAttrib:
            
                # Load original values results
                attr = np.load(pathLoad + pathSG)
                
                # Convert SG maps to absolute (not SGSQ)
                if ('SG+' in pathSG):
                    # Convert to absolute values
                    attr = np.abs(attr)
                
                # Save absolute values results
                os.makedirs(pathAbsoluteAttr + method + '/', exist_ok=True)
                np.save(pathAbsoluteAttr + method + '/' + pathSG, attr)

            
# Using the special variable
if __name__=="__main__": 
    main()