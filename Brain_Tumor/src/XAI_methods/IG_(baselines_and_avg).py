# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:32:28 2021

@author: Wistan
"""


import os
import torch
import numpy as np
# import pickle
import saliency.core as saliency
import logging
import sys
from monai.data import Dataset
from monai.data import list_data_collate
from torch.utils.data import DataLoader
import torch.backends.cudnn
import nibabel as nib

from utils_interpret import parse_args, get_models_from_entry, create_patient_dict_classification, create_dataset_files, get_transforms, get_model, do_pred, shape_to_model
from classif_eval_interpret import create_parser, save_results_fold, do_inference


# Base paths
pathRoot = './'
pathResults = pathRoot + 'Results/Raw_attrs/IG'



def standardize_image(im):
    extreme_value = np.max(np.abs(im))
    return im / extreme_value



def call_model_function(images, call_model_args=None, expected_keys=None):
    device = call_model_args[0]
    model = call_model_args[1]
    target = call_model_args[2]
    
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    output = do_pred(models=model, x=images, sigmoid=False, softmax=True)
    
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:, target]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))[0].squeeze(axis=1)
        gradients = grads.cpu().detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

    



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
        
        # Empty attribution arrays to add each model's saliency map
        attrs_min = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_zero = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_max = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_min_zero = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_min_max = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_zero_max = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        attrs_mean = np.zeros_like(val_inputs.squeeze().cpu().detach().numpy(), dtype=np.float64)
        model_iter = 0
        
        # Load input image and raw attributions for given slice and method
        val_inputs_numpy = val_inputs.squeeze().cpu().detach().numpy()
        
        # Baseline values are zero / min of input / max of input
        baseline_min = np.full(val_inputs_numpy.shape, np.min(val_inputs_numpy))
        baseline_zero = np.zeros_like(val_inputs_numpy, dtype=np.float64)
        baseline_max = np.full(val_inputs_numpy.shape, np.max(val_inputs_numpy))
        
        for model_to_load in config_args_list:
            model = get_model(model_type=model_to_load.net['model'], model_params=model_to_load.net['params'])
            model.load_state_dict(torch.load(model_to_load.model_pth, map_location=device), strict=False)

            model = model.to(device)
            model.eval()
            
            val_inputs, p_b, p_a = shape_to_model(val_inputs, min_shape=48)
            
            ### IG BEGINS
            
            print('\t\t Launch IG on model', model_iter+1)
            
            # Construct the saliency object. This alone doesn't do anything.
            integrated_gradients = saliency.IntegratedGradients()
        
            # Compute Integrated Gradients for each baseline (ideally 200 steps, test for batch size)
            attr_min = integrated_gradients.GetMask(x_value=val_inputs_numpy, call_model_function=call_model_function, call_model_args=(device, model, target[step]), x_baseline=baseline_min, x_steps=200, batch_size=5)
            attr_zero = integrated_gradients.GetMask(x_value=val_inputs_numpy, call_model_function=call_model_function, call_model_args=(device, model, target[step]), x_baseline=baseline_zero, x_steps=200, batch_size=5)
            attr_max = integrated_gradients.GetMask(x_value=val_inputs_numpy, call_model_function=call_model_function, call_model_args=(device, model, target[step]), x_baseline=baseline_max, x_steps=200, batch_size=5)
            
            # Standardize saliency maps for average
            attr_min_stand = standardize_image(attr_min)
            attr_zero_stand = standardize_image(attr_zero)
            attr_max_stand = standardize_image(attr_max)
            
            # Averages 2-by-2, and overall
            attr_min_zero = np.mean([attr_min_stand, attr_zero_stand], axis=0)
            attr_min_max = np.mean([attr_min_stand, attr_max_stand], axis=0)
            attr_zero_max = np.mean([attr_zero_stand, attr_max_stand], axis=0)
            attr_mean = np.mean([attr_min_stand, attr_zero_stand, attr_max_stand], axis=0)
            
            print('\t\t End IG on model', model_iter+1)
            
            # # Get the model name
            # model_name = model_to_load.model_pth[model_to_load.model_pth.find('\\')+1 :]
            # model_name = model_name[: model_name.rfind('.')]
            # model_name = model_name.replace('\\', '_')

            # # Save attributions as .nii file
            # ni_img_black = nib.Nifti1Image(attr_black, affine=val_data['input_meta_dict']['affine'].squeeze())
            # nib.save(ni_img_black, pathResults + '(Black)/' + val_data['patient_id'][0] + '_' + model_name + ".nii")
            # ni_img_white = nib.Nifti1Image(attr_white, affine=val_data['input_meta_dict']['affine'].squeeze())
            # nib.save(ni_img_white, pathResults + '(White)/' + val_data['patient_id'][0] + '_' + model_name + ".nii")
            # ni_img_mean = nib.Nifti1Image(attr_mean, affine=val_data['input_meta_dict']['affine'].squeeze())
            # nib.save(ni_img_mean, pathResults + '(BlackWhite)/' + val_data['patient_id'][0] + '_' + model_name + ".nii")
            
            # Prepare for average attribution
            attrs_min += attr_min
            attrs_zero += attr_zero
            attrs_max += attr_max
            attrs_min_zero += attr_min_zero
            attrs_min_max += attr_min_max
            attrs_zero_max += attr_zero_max
            attrs_mean += attr_mean
            
            model_iter += 1
            
            ### IG ENDS
            
        # Average of attributions
        attrs_min /= model_iter
        attrs_zero /= model_iter
        attrs_max /= model_iter
        attrs_min_zero /= model_iter
        attrs_min_max /= model_iter
        attrs_zero_max /= model_iter
        attrs_mean /= model_iter
        
        # Save saliency maps (averaged over models)
        ni_attrs_min = nib.Nifti1Image(attrs_min, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_zero = nib.Nifti1Image(attrs_zero, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_max = nib.Nifti1Image(attrs_max, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_min_zero = nib.Nifti1Image(attrs_min_zero, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_min_max = nib.Nifti1Image(attrs_min_max, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_zero_max = nib.Nifti1Image(attrs_zero_max, affine=val_data['input_meta_dict']['affine'].squeeze())
        ni_attrs_mean = nib.Nifti1Image(attrs_mean, affine=val_data['input_meta_dict']['affine'].squeeze())
        
        os.makedirs(pathResults + '(min)/', exist_ok=True)
        os.makedirs(pathResults + '(0)/', exist_ok=True)
        os.makedirs(pathResults + '(max)/', exist_ok=True)
        os.makedirs(pathResults + '(min-0)/', exist_ok=True)
        os.makedirs(pathResults + '(min-max)/', exist_ok=True)
        os.makedirs(pathResults + '(0-max)/', exist_ok=True)
        os.makedirs(pathResults + '(min-0-max)/', exist_ok=True)
        
        nib.save(ni_attrs_min, pathResults + '(min)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_zero, pathResults + '(0)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_max, pathResults + '(max)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_min_zero, pathResults + '(min-0)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_min_max, pathResults + '(min-max)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_zero_max, pathResults + '(0-max)/' + val_data['patient_id'][0] + ".nii")
        nib.save(ni_attrs_mean, pathResults + '(min-0-max)/' + val_data['patient_id'][0] + ".nii")

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
    # save_results_fold(y=y, y_prob=y_prob, y_pred=y_pred, patients_id=patients_id,
    #                   save_path=parser_of_args.save_path)
    
    do_interpretability(config_args_list, target=y_pred)


if __name__ == "__main__":
    sys.exit(main_process(sys.argv[1:]))

