# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:05:00 2021

@author: Wistan
"""


import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.nn import max_pool_with_argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
import math
import matplotlib.pyplot as plt

pathRoot = './'
pathImages = pathRoot + 'Input Images/'
pathResults = pathRoot + 'Results/Raw_attrs/Deconvolution/'
pathRawAttr = pathRoot + 'Results/Methods_Attributions_Inversed/'



# ####### CONVOLUTION NETWORK (to be mirrored)

# model = Sequential()
# model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(512, 512, 3)))
# model.add(Conv2D(8, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.50))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

# ############################



def intermediary_maps(model, image):
    
    for i in range(12):
        # Code sniPpet for Intermediary Maps
        with tf.device('cpu'):
            model.layers[i]._name = str(i)
            layer_output=model.get_layer(str(i)).output
            intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
            pred=intermediate_model.predict(image)[0]
        
        display_intermediary_maps(pred, i, 'Conv')
        


def display_intermediary_maps(maps, layerIdx, mapType):
        
    nbDpi = 72
    space_pix = int(0.15 * maps.shape[0])
    decal_colorbar = 150
    square_dim = (int)(math.ceil(math.sqrt(maps.shape[-1])))
    fig_width = (maps.shape[1] * square_dim + space_pix * (square_dim+1)) / nbDpi
    fig_height = (maps.shape[0] * square_dim + space_pix * (square_dim+1) + decal_colorbar) / nbDpi
    
    cursor_x = space_pix
    cursor_y = space_pix + decal_colorbar
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    for idx in range(maps.shape[-1]):
        fig.figimage(maps[:, :, idx], xo=cursor_x, yo=cursor_y, origin='upper')
        if (idx%square_dim == (square_dim-1)):
            cursor_x = space_pix
            cursor_y += maps.shape[0] + space_pix
        else:
            cursor_x += maps.shape[0] + space_pix
    
    fig.savefig('D:/PENDING WORK/Results/Layer_' +mapType+ '_' +str(layerIdx)+ '_viridis.tiff', facecolor=fig.get_facecolor(), edgecolor='none', dpi=72)



def create_deconv():

    deflatten = Reshape((60, 60, 32), input_shape=(115200,))

    deconv_block1_1 = Conv2DTranspose(filters=32, kernel_size=3, input_shape=(121, 121, 32))
    deconv_block1_2 = Conv2DTranspose(filters=16, kernel_size=3, input_shape=(123, 123, 32), use_bias=False)

    deconv_block2_1 = Conv2DTranspose(filters=16, kernel_size=3, input_shape=(250, 250, 16))
    deconv_block2_2 = Conv2DTranspose(filters=8, kernel_size=3, input_shape=(252, 252, 16), use_bias=False)

    deconv_block3_1 = Conv2DTranspose(filters=8, kernel_size=3, input_shape=(508, 508, 8))
    deconv_block3_2 = Conv2DTranspose(filters=3, kernel_size=3, input_shape=(510, 510, 8), use_bias=False)

    flatten_model = Sequential()
    flatten_model.add(deflatten)
    
    deconv1_model = Sequential()
    deconv1_model.add(deconv_block1_1)
    deconv1_model.add(deconv_block1_2)

    deconv2_model = Sequential()
    deconv2_model.add(deconv_block2_1)
    deconv2_model.add(deconv_block2_2)

    deconv3_model = Sequential()
    deconv3_model.add(deconv_block3_1)
    deconv3_model.add(deconv_block3_2)
    
    return flatten_model, deconv1_model, deconv2_model, deconv3_model



def load_model(deconv1_model, deconv2_model, deconv3_model, original_model):

    deconv1_model.layers[0].set_weights(original_model.layers[9].get_weights())
    deconv1_model.layers[1].set_weights([original_model.layers[8].get_weights()[0]])
    deconv2_model.layers[0].set_weights(original_model.layers[5].get_weights())
    deconv2_model.layers[1].set_weights([original_model.layers[4].get_weights()[0]])
    deconv3_model.layers[0].set_weights(original_model.layers[1].get_weights())
    deconv3_model.layers[1].set_weights([original_model.layers[0].get_weights()[0]])
    
    return deconv1_model, deconv2_model, deconv3_model
    


def get_activations(image, model, layerIdx, maxpool=True):
    
    # Results before maxpool
    model.layers[layerIdx]._name = 'output_layer'
    output_layer = model.get_layer('output_layer').output
    intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=output_layer)
    model.layers[layerIdx]._name = str(layerIdx)
    with tf.device('cpu'):
        inter_pred = intermediate_model.predict(image)

    # Results after custom maxpool
    if (maxpool):
        output, argmax = max_pool_with_argmax(inter_pred, ksize=(2, 2), strides=2, padding='VALID')
        output = output.numpy()[0]
        argmax = argmax.numpy()[0]
        return inter_pred.shape, argmax
    else:
        return inter_pred
    
    
    
def custom_unpooling(output_shape, switch_indices, activations):
    # Build Unpooled map from maxpool output and switch indices array
    unpooled_map = np.zeros(output_shape)

    for channel in range(unpooled_map.shape[-1]):
        a = ((switch_indices[:, :, channel] - channel) / unpooled_map.shape[-1]).astype(int)
        y_arr = a // output_shape[1]
        x_arr = a % output_shape[1]

        for y_idx in range(y_arr.shape[0]):
            for x_idx in range(x_arr.shape[-1]):
                y_value = y_arr[y_idx, x_idx]
                x_value = x_arr[y_idx, x_idx]
                unpooled_map[0, y_value, x_value, channel] = activations[0, y_idx, x_idx, channel]
                
    return unpooled_map
    


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
    
        # Define device
        device = tf.device('cpu')
        
        #######     WEIGHTS TRANSFER       #######
        
        # Load model
        model = tf.keras.models.load_model(pathRoot + 'serialized_model', compile=False)
        
        # DeconvNet instance
        flatten_model, deconv1_model, deconv2_model, deconv3_model = create_deconv()
        
        # Weights transfer to DeconvNet
        deconv1_model, deconv2_model, deconv3_model = load_model(deconv1_model, deconv2_model, deconv3_model, model)
        
        # #######     FULL DECONVOLUTION ATTRIBUTION MAP       #######
        
        # For each input image we display and save the results
        for imageIdx in range (len(images_paths)):
        
            print("\t Deconvolution Image number", imageIdx+1)
            
            # Load and process input image
            im_numpy = cv2.imread(dir_path + images_paths[imageIdx])
            im_numpy = cv2.resize(im_numpy, (512, 512))
            im_numpy = im_numpy.astype('uint8') / 255
            im_numpy = np.expand_dims(im_numpy, axis=0)
            im_tensor = tf.convert_to_tensor(im_numpy)
        
            # Get switch indices for all maxpool and activations after flatten
            output_shape_1, switch_indices_1 = get_activations(im_tensor, model, 1, maxpool=True)
            output_shape_2, switch_indices_2 = get_activations(im_tensor, model, 5, maxpool=True)
            output_shape_3, switch_indices_3 = get_activations(im_tensor, model, 9, maxpool=True)
            activations = get_activations(im_tensor, model, 12, maxpool=False)
            
            # Launch deconv model part by part
            with device:
                # Deflattening
                deflattened = flatten_model(activations)
                # Unpooling number 1
                unpooled_1 = custom_unpooling(output_shape_3, switch_indices_3, deflattened)
                # Deconv number 1
                deconv_1 = deconv1_model(unpooled_1)
                # Unpooling number 2
                unpooled_2 = custom_unpooling(output_shape_2, switch_indices_2, deconv_1)
                # Deconv number 2
                deconv_2 = deconv2_model(unpooled_2)
                # Unpooling number 3
                unpooled_3 = custom_unpooling(output_shape_1, switch_indices_1, deconv_2)
                # Deconv number 3
                deconv_3 = deconv3_model(unpooled_3)
                deconv_3_numpy = deconv_3.numpy()[0]
                
                # # For Intermediary maps Visualization
                # intermediary_maps(model, im_tensor)
                # deflattened_numpy = deflattened.numpy()[0]
                # display_intermediary_maps(deflattened_numpy, 12, 'Deconv')
                # unpooled_1_numpy = unpooled_1[0]
                # display_intermediary_maps(unpooled_1_numpy, 10, 'Deconv')
                # deconv_1_numpy = deconv_1.numpy()[0]
                # display_intermediary_maps(deconv_1_numpy, 8, 'Deconv')
                # unpooled_2_numpy = unpooled_2[0]
                # display_intermediary_maps(unpooled_2_numpy, 6, 'Deconv')
                # deconv_2_numpy = deconv_2.numpy()[0]
                # display_intermediary_maps(deconv_2_numpy, 4, 'Deconv')
                # unpooled_3_numpy = unpooled_3[0]
                # display_intermediary_maps(unpooled_3_numpy, 2, 'Deconv')
                # display_intermediary_maps(deconv_3_numpy, 0, 'Deconv')
                
            # Take maximum across channels
            heatmap = np.amax(deconv_3_numpy, axis=-1)
            
            # Raw attributions save
            np.save(pathResults + dirs[dirIdx] + '/Raw_Deconvolution_' +images_paths[imageIdx]+ '.npy', heatmap)



# Using the special variable
if __name__ == "__main__":
    main()

