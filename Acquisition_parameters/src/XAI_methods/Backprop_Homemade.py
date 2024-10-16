# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:32:28 2021

@author: Wistan
"""

import torch
import numpy as np
import pickle


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Raw_attrs/'
methodFolder = 'BP/'


# Defining main function
def main(): 
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = pickle.load(open(pathRoot + 'serialized_param_predictor.sav', 'rb')).to(device)
    
    # Images chosen for application of saliency maps
    test_images = torch.load(pathRoot + 'test_slices_263.pt').to(device)
    # Load params dictionnary
    f = open("./params.pkl", "rb")
    params = pickle.load(f)
    f.close()
    
    
    #######     FULL BACKPROPAGATION ATTRIBUTION MAP       #######
    
    
    # For each input image we display and save the results
    for i in range (len(test_images)):
        
        print("Backpropagation Slice number", i+1)
        
        # Load input image
        im_tensor = test_images[i].unsqueeze(0).to(device)
        
        # Set the requires_grad_ to the image for retrieving gradients
        im_tensor.requires_grad = True
    
        # Retrieve output from the image
        y_pred = model(im_tensor)
    
        # For each parameter prediction
        for param_name, pred in y_pred.items():
            
            # Apply sigmoid on branch prediction
            pred = torch.sigmoid(pred)
            # Find maximum, index, and class of prediction
            idx_max_pred = pred.argmax()
            max_pred = pred[0, idx_max_pred]
            pred_class = params[param_name][idx_max_pred.detach().cpu().numpy()]
    
            # Do backpropagation to get the derivative of the output based on the image
            max_pred.backward(retain_graph=True)
            
            # Retrieve the saliency map
            attribution_map = im_tensor.grad.data.squeeze().cpu().numpy()
            
            # Raw attribution save
            if (param_name == 'contrast_used'):
                np.save(pathResults + param_name + '/' + methodFolder + 'BP_Im_' +str(i+1)+ '_pred_' + str(max_pred.detach().cpu().numpy()) + '.npy', attribution_map)
            else:
                np.save(pathResults + param_name + '/' + methodFolder + 'BP_Im_' +str(i+1)+ '_pred_' + str(pred_class) + '.npy', attribution_map)
        

# Using the special variable
if __name__=="__main__": 
    main()