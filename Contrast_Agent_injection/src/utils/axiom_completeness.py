# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:44:33 2022

@author: Wistan
"""


import torch
import numpy as np
import pickle


# Base paths
pathRoot = './'
pathRawAttrBlack = pathRoot + 'Results/Raw_attrs/IntegratedGradients(Black)/'
pathRawAttrWhite = pathRoot + 'Results/Raw_attrs/IntegratedGradients(White)/'
pathRawAttrBW = pathRoot + 'Results/Raw_attrs/IntegratedGradients(BlackWhite)/'
pathRawAttrEG = pathRoot + 'Results/Raw_attrs/ExpectedGradients/'

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = pickle.load(open('./serialized_resnet.sav', 'rb')).to(device)

# Background slices load
background = torch.load(pathRoot + 'baseline_200.pt').to(device)
# Load images tensor
test_images = torch.load(pathRoot + 'test_slices_290.pt').to(device)
# Load labels tensor
test_labels = torch.load(pathRoot + 'test_labels_290.pt').numpy()
    
nsamples=200
nb_test_slices = 290
nb_runs = 15



#####################                 IG Completeness axiom Validation                 #####################



sum_attrs_black = np.zeros(nb_test_slices, dtype=float)
sum_attrs_white = np.zeros(nb_test_slices, dtype=float)
sum_attrs_bw = np.zeros(nb_test_slices, dtype=float)
input_pred = np.zeros(nb_test_slices, dtype=float)

# Input image as numpy array
im_numpy = test_images[0].squeeze().cpu().numpy()

# Baselines are both black and white images
baseline_black = torch.from_numpy(np.zeros_like(im_numpy)).float().unsqueeze(0).unsqueeze(0).to(device)
baseline_white = torch.from_numpy(np.ones_like(im_numpy)).float().unsqueeze(0).unsqueeze(0).to(device)

# Apply model to baselines and input
ref_pred_black = torch.sigmoid(model(baseline_black)).detach().cpu().numpy()[0][0]
ref_pred_white = torch.sigmoid(model(baseline_white)).detach().cpu().numpy()[0][0]
ref_pred_bw = (ref_pred_black + ref_pred_white) / 2

# For each input image we display and save the results (IG results)
for sliceIdx in range (nb_test_slices):
        
    print("Slice number", sliceIdx+1)
    
    # Load raw original attributions
    raw_attrs_black = np.load(pathRawAttrBlack + 'Raw_attr_IntegratedGradients(Black)_Im_' +str(sliceIdx+1)+ '.npy')
    raw_attrs_white = np.load(pathRawAttrWhite + 'Raw_attr_IntegratedGradients(White)_Im_' +str(sliceIdx+1)+ '.npy')
    raw_attrs_bw = np.load(pathRawAttrBW + 'Raw_attr_IntegratedGradients(BlackWhite)_Im_' +str(sliceIdx+1)+ '.npy')
    input_pred[sliceIdx] = torch.sigmoid(model(test_images[sliceIdx].unsqueeze(0))).detach().cpu()
    
    if (test_labels[sliceIdx] <= 0.5):
        raw_attrs_black = raw_attrs_black * -1
        raw_attrs_white = raw_attrs_white * -1
        raw_attrs_bw = raw_attrs_bw * -1
    
    # Sum of attributions
    sum_attrs_black[sliceIdx] = np.sum(raw_attrs_black)
    sum_attrs_white[sliceIdx] = np.sum(raw_attrs_white)
    sum_attrs_bw[sliceIdx] = np.sum(raw_attrs_bw)
    
diff_input_ref_black = input_pred - ref_pred_black
diff_input_ref_white = input_pred - ref_pred_white
diff_input_ref_bw = input_pred - ref_pred_bw

percentage_value_black = np.zeros(nb_test_slices, dtype=float)
percentage_value_white = np.zeros(nb_test_slices, dtype=float)
percentage_value_bw = np.zeros(nb_test_slices, dtype=float)

for i in range (nb_test_slices):
    percentage_value_black[i] = round(min(sum_attrs_black[i]/diff_input_ref_black[i], diff_input_ref_black[i]/sum_attrs_black[i]) * 100, 1)
    percentage_value_white[i] = round(min(sum_attrs_white[i]/diff_input_ref_white[i], diff_input_ref_white[i]/sum_attrs_white[i]) * 100, 1)
    percentage_value_bw[i] = round(min(sum_attrs_bw[i]/diff_input_ref_bw[i], diff_input_ref_bw[i]/sum_attrs_bw[i]) * 100, 1)



#####################                 EG Completeness axiom Validation                 #####################



list_background = [background]

mean_ref_pred = np.zeros(nb_test_slices, dtype=float)

input_pred = np.zeros(nb_test_slices, dtype=float)
# Apply model on chosen test slices
for sliceIdx in range (nb_test_slices):
    input_pred[sliceIdx] = model(test_images[sliceIdx].unsqueeze(0)).detach().cpu().numpy()[0][0]

bkgd_preds = []
# Apply model on all background slices
for bkgdIdx in range (len(background)):
    bkgd_preds.append(model(background[bkgdIdx].unsqueeze(0)).detach().cpu().numpy()[0][0])

# For each run (random seed)
for rseed in range (nb_runs):
    print("Expected Gradients Run number", rseed+1)

    np.random.seed(rseed)
    
    for sliceIdx in range (nb_test_slices):
                
        predictions = []
    
        # fill in the samples arrays
        for k in range(nsamples):
            rind = np.random.choice(list_background[0].shape[0])
            t = np.random.uniform()
            # Get prediction of randomly chosen slice
            predictions.append(bkgd_preds[rind])
            
        # Average of references output predictions
        mean_ref_pred[sliceIdx] += np.mean(predictions)
    
# Divide by number of runs
mean_ref_pred /= nb_runs
# Difference between output of image and reference
diff_input_ref = input_pred - mean_ref_pred

# Attributions sum and percentage of difference arrays
sum_attrs = np.zeros(nb_test_slices, dtype=float)
percentage_value = np.zeros(nb_test_slices, dtype=float)

for i in range(nb_test_slices):
    attrs = np.load(pathRawAttrEG + "Raw_attr_ExpectedGradients_Im_{slice}.npy".format(slice=i+1))
    if (test_labels[i] <= 0.5):
        attrs = attrs * -1
    sum_attrs[i] = np.sum(attrs)
    
    percentage_value[i] = round(min(sum_attrs[i]/diff_input_ref[i], diff_input_ref[i]/sum_attrs[i]) * 100, 1)