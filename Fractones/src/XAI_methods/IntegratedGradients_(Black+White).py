# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import os
import numpy as np
import cv2
import tensorflow as tf
import saliency.core as saliency


# Base paths
pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathResults = pathRoot + 'Results/Raw_attrs/IntegratedGradients'
pathRawAttr = pathRoot + 'Results/Methods_Attributions_Inversed/'

# Define device
device = tf.device('cpu')

# Load model
model = tf.keras.models.load_model(pathRoot + 'serialized_model', compile=False)


def call_model_function(images, call_model_args=None, expected_keys=None):
    im_tensor = tf.convert_to_tensor(images)
    # Launch model (to cpu)
    with device:
        with tf.GradientTape() as tape:
            tape.watch(im_tensor)
            output = model(im_tensor)
    
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            # Compute gradients
            grads = tape.gradient(output, im_tensor)
            # Convert to numpy
            gradients = grads.numpy()
        
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}


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
        
        
        #######     FULL INTEGRATED GRADIENTS ATTRIBUTION MAP       #######
        
        
        # For each input image we display and save the results
        for imageIdx in range (len(images_paths)):
            
            print("\t Integrated Gradients Image number", imageIdx+1)
            
            # Load and process input image
            im_numpy = cv2.imread(dir_path + images_paths[imageIdx])
            im_numpy = cv2.resize(im_numpy, (512, 512))
            im_numpy = im_numpy.astype('uint8') / 255
            
            # Construct the saliency object. This alone doesn't do anything.
            integrated_gradients = saliency.IntegratedGradients()
            
            # Baselines are both black and white images
            baseline_black = np.zeros_like(im_numpy)
            baseline_white = np.ones_like(im_numpy)
            
            # Compute Integrated Gradients for each baseline
            attr_black = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, x_baseline=baseline_black, x_steps=200, batch_size=25)
            attr_white = integrated_gradients.GetMask(x_value=im_numpy, call_model_function=call_model_function, x_baseline=baseline_white, x_steps=200, batch_size=25)
            # Create a combined attribution map (average of black and white)
            attrs = [attr_black, attr_white]
            attrs_mean = np.mean(attrs, axis=0)
            # Take maximum across channels
            attr_black = np.amax(attr_black, axis=-1)
            attr_white = np.amax(attr_white, axis=-1)
            attrs_mean = np.amax(attrs_mean, axis=-1)
            
            # Raw attributions save
            np.save(pathResults + '(Black)/' + dirs[dirIdx] + '/Raw_IntegratedGradients(Black)_' +images_paths[imageIdx]+ '.npy', attr_black)
            np.save(pathResults + '(White)/' + dirs[dirIdx] + '/Raw_IntegratedGradients(White)_' +images_paths[imageIdx]+ '.npy', attr_white)
            np.save(pathResults + '(BlackWhite)/' + dirs[dirIdx] + '/Raw_IntegratedGradients(BlackWhite)_' +images_paths[imageIdx]+ '.npy', attrs_mean)
            
            

# Using the special variable
if __name__=="__main__": 
    main()