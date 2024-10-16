# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:03:00 2022

@author: Wistan
"""

import torch
import pickle

from xception import Xception as Model


pathSource = "./"
pathWeights = pathSource + 'Xception_pedro/Trained_model/model_weights.pth'


# Defining Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = Model(pretrained=True, num_classes=1).to(device)
model.load_state_dict(torch.load(pathWeights, map_location=device))
model.eval()

# Save the model to disk
filename = "serialized_Xception.sav"
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))