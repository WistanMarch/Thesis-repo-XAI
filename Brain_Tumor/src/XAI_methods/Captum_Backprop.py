# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:32:28 2021

@author: Wistan
"""

import torch
import numpy as np
import logging
import sys
from monai.data import Dataset
from monai.data import list_data_collate
from torch.utils.data import DataLoader
import torch.backends.cudnn
import nibabel as nib
from captum.attr import Saliency

from utils_interpret import parse_args, get_models_from_entry, create_patient_dict_classification, create_dataset_files, get_transforms, get_model, do_pred
from utils_interpret import shape_to_model
from classif_eval_interpret import save_results_fold, create_parser, do_inference


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Raw_attrs/Captum_Backpropagation/'



def do_interpretability(config_args_list, target):
    logger = logging.getLogger('Inference')

    gen_args = config_args_list[0]  # use the args from the first list as parser args  are all duplicate for each config

    input_path = gen_args.data_folder['input_path']
    label_path = None
    labels_name = None

    # Get files for inference
    if gen_args.mode in ['val', 'val_native', 'train']:
        label_path = gen_args.data_folder['label_path']
        labels_name = gen_args.data_folder['label']['label_col']
    else:
        raise ValueError('mode for inference need to be train, val_native, val or test')

    # creation of dict for inference
    patient_dict = create_patient_dict_classification(input_path=input_path,
                                                      label_path=label_path,
                                                      modality=gen_args.data_folder['modality'],
                                                      **gen_args.data_folder['label'])

    # set patient for inference with correct transform
    val_patient = []
    val_patient = [*patient_dict]

    # create dataset file for MONAI
    val_files = create_dataset_files(patient_dict, val_patient, gen_args.data_folder['modality'], labels_name)

    # get transforms to apply
    _, val_transforms = get_transforms(train_transforms=gen_args.transforms['train_transforms'],
                                        val_transforms=gen_args.transforms['val_transforms'])

    # Get DataSet loader type
    val_ds = Dataset(data=val_files, transform=val_transforms)
    # Create a validation loader
    val_loader = DataLoader(val_ds, **gen_args.val["params"], collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available())

    # do inference only on one GPU, if it was set on multiple in the config
    if isinstance(gen_args.gpu, list):
        gen_args.gpu = gen_args.gpu[0]

    # Set device
    torch.cuda.set_device(gen_args.gpu)
    device = torch.device(f'cuda:{gen_args.gpu}')

    # init vars for pred
    patient_id = []
    
    for step, val_data in enumerate(val_loader):
        logger.info(f"Step: {step + 1}/{len(val_loader)} - Running on {val_data['patient_id']}")
        patient_id.append(val_data['patient_id'])

        # Now put in GPU
        val_inputs = val_data['input'].to(device)
        val_inputs.requires_grad = True
        
        # Here models are load each time much slower but reduce memory consumption by storing all models on GPU or by swapping CPU/GPU
        attrs = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
    
        for model_to_load in config_args_list:
            model = get_model(model_type=model_to_load.net['model'], model_params=model_to_load.net['params'])
            model.load_state_dict(torch.load(model_to_load.model_pth, map_location=device), strict=False)

            model = model.to(device)
            model.eval()
            
            val_inputs, p_b, p_a = shape_to_model(val_inputs, min_shape=48)
            
            ### BP BEGINS
            
            # Backpropagation init
            saliency = Saliency(do_pred)
            
            # Computes saliency maps
            attribution_map = saliency.attribute(inputs=val_inputs, target=target[step], abs=False, additional_forward_args=(model, False, True))
            attr_numpy = attribution_map.squeeze().detach().cpu().numpy()
                    
            # Get the model name
            model_name = model_to_load.model_pth[model_to_load.model_pth.find('\\')+1 :]
            model_name = model_name[: model_name.rfind('.')]
            model_name = model_name.replace('\\', '_')

            # Save attributions as .nii file
            ni_img = nib.Nifti1Image(attr_numpy, affine=val_data['input_meta_dict']['affine'].squeeze())
            nib.save(ni_img, pathResults + val_data['patient_id'][0] + '_' + model_name + ".nii")
            
            ### BP ENDS
            
            # Prepare for average attribution
            attrs += attr_numpy

        # Average of attributions
        attrs /= 25
        # Save average
        ni_attrs = nib.Nifti1Image(attrs, affine=val_data['input_meta_dict']['affine'].squeeze())
        nib.save(ni_attrs, pathResults + val_data['patient_id'][0] + ".nii")

        # empty cache
        torch.cuda.empty_cache()


def main_process(args=None):
    
    parser_of_args = create_parser().parse_args(args)

    if parser_of_args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif parser_of_args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    # try:
    model_dict = get_models_from_entry(parser_of_args.config_file)
    # create list of Namespace. Each element correspond to the config of each weight
    # Furthermore update the config file with set_change. Not optimal, was fast coded to make operable
    # the original code
    config_args_list = []
    for yaml_config_file, model_pth in model_dict.items():
        config_for_weight = create_parser().parse_args()
        config_for_weight.config_file = yaml_config_file
        config_for_weight.model_pth = model_pth
        args = parse_args(config_for_weight)
        config_args_list.append(args)

    patients_id, y, y_prob, y_pred = do_inference(config_args_list)
    save_results_fold(y=y, y_prob=y_prob, y_pred=y_pred, patients_id=patients_id,
                      save_path=parser_of_args.save_path)
    
    do_interpretability(config_args_list, target=y_pred)


if __name__ == "__main__":
    sys.exit(main_process(sys.argv[1:]))