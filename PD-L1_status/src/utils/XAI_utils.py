#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from monai.transforms import Compose, EnsureType
from monai.transforms import EnsureChannelFirst, Resize, ScaleIntensity, NormalizeIntensity
from monai.data import ImageDataset, DataLoader
from utils.manage_immugast_XAI import immugast_split
from utils.exam_immugast import exam_immugast
from utils.whichnet import whichnet_3D
import numpy as np
import torch



# Return transforms for validation & test sets
def get_transforms(dataset, size):
    
    if dataset == 'immugast':
        transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((size, size, int(size/2))), EnsureType(), NormalizeIntensity(nonzero=True, channel_wise=False),])
    
    return transforms



# Construct dataset & return useful elements for XAI
def XAI_dataset(params, trial, device, loading):
    
    # IDs of validation & test patients
    val_ids, test_ids = immugast_split(params['feature'], params['cutoff'], trial)
    ids = []
    
    # Validation & test transforms are identical, we get the test transforms for interpretability
    infer_transforms = get_transforms('immugast', params['size'])
    
    # Labels & files names for inference
    imgs, labels = list(), list()
    
    # Iterate on validation subset
    for val_id in val_ids:
        exam = exam_immugast(int(val_id), loading, upload=False, offsetx=params['offsetx'], offsety=params['offsety'], offsetz=params['offsetz'])
        imgs.append(exam.folder+exam.id+'.nii.gz')
        ids.append(val_id)
        if params['feature'] == 'CPS':
            exam.binarize_CPS(params['cutoff'])
            labels.append(int(exam.CPSb))
    
    # Iterate on test subset
    for test_id in test_ids:
        exam = exam_immugast(int(test_id), loading, upload=False, offsetx=params['offsetx'], offsety=params['offsety'], offsetz=params['offsetz'])
        imgs.append(exam.folder+exam.id+'.nii.gz')
        ids.append(test_id)
        if params['feature'] == 'CPS':
            exam.binarize_CPS(params['cutoff'])
            labels.append(int(exam.CPSb))
    
    # Array of labels
    labels = np.array(labels, dtype=int)
    
    # Prepare the dataset
    ds = ImageDataset(image_files=imgs, labels=labels, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=params['batch'], shuffle=False)
    
    # Load model (+ eval mode)
    net = whichnet_3D(params['net_id'], device, params['size'], True)
    net.load_state_dict(torch.load(loading+'t-'+str(trial)+'-model.pth', map_location=device))
    net.eval()
    
    # Return
    return ids, labels, loader, net