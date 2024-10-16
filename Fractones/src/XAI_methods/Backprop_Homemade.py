# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:32:28 2021

@author: Wistan
"""

import os
import numpy as np
import cv2
import tensorflow as tf


# Base paths
pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathResults = pathRoot + 'Results/Raw_attrs/Backpropagation/'
pathRawAttr = pathRoot + 'Results/Methods_Attributions_Inversed/'

# Define device
device = tf.device('cpu')

# Load model
model = tf.keras.models.load_model(pathRoot + 'serialized_model', compile=False)




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
        
        
        #######     FULL BACKPROPAGATION ATTRIBUTION MAP       #######
    
    
        # For each input image we display and save the results
        for imageIdx in range (len(images_paths)):
        
            print("\t Backpropagation Image number", imageIdx+1)
        
            # Load and process input image
            im_numpy = cv2.imread(dir_path + images_paths[imageIdx])
            im_numpy = cv2.resize(im_numpy, (512, 512))
            im_numpy = im_numpy.astype('uint8') / 255
            im_numpy = np.expand_dims(im_numpy, axis=0)
            im_tensor = tf.convert_to_tensor(im_numpy)
        
            # Launch model (to cpu)
            with device:
                with tf.GradientTape() as tape:
                    tape.watch(im_tensor)
                    y_pred = model(im_tensor)
        
                # Compute gradients
                grads = tape.gradient(y_pred, im_tensor)
        
            # Take maximum across channels
            gradient = tf.reduce_max(grads, axis=-1)
        
            # Convert to numpy
            gradient = gradient.numpy()
        
            # Raw attribution save
            np.save(pathResults + dirs[dirIdx] + '/Raw_Backpropagation_' +images_paths[imageIdx]+ '.npy', gradient[0])



# Using the special variable
if __name__=="__main__": 
    main()