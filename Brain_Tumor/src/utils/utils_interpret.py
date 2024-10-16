# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE
| Created on: April, 2020
"""
import os
import GPUtil as gpu
import re
from glob import glob
import oyaml as yaml  # oyaml is a drop-in replacement for PyYAML which preserves dict ordering.
from collections import abc
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import KFold, StratifiedKFold
import logging
import copy
from monai import transforms
from monai.transforms import Compose
import networks
import torch
import torch.nn.functional as F
# from tta import simple_tta

logger = logging.getLogger(__name__)


def split_filename(file_name):
    """
    Split file_name into folder path name, basename, and extension name.
    :param file_name: full path
    :return: path name, basename, extension name
    """
    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in ['.nii.gz']:
        ext_len = len(special_ext)
        if fname[-ext_len:].lower() == special_ext:
            ext = fname[-ext_len:]
            fname = fname[:-ext_len] if len(fname) > ext_len else ''
            break
    if not ext:
        fname, ext = os.path.splitext(fname)
    return pth, fname, ext


def list_of_gpu():
    """
    Check for gpu ID
    :return: return list of ID of GPUs
    """
    return [gpu_instance.id for gpu_instance in gpu.getGPUs()]


def update_nested_dict(d, u):
    """
    Update a nested dict
    :param d: dict to update
    :param u: dict of key and value to update
    :return: the d dict updated
    """
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_args(args):
    """
    parser of config file. Allow the overwriting of arguments after reading a config file
    :param args: args
    :return: args
    """
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.SafeLoader)
        arg_dict = args.__dict__

        # remove top level and add as args
        if args.set_change is not None:
            for key, value in args.set_change.items():
                setattr(args, key, value)
            delattr(args, 'set_change')

        # update
        for key, value in data.items():
            if key in arg_dict and arg_dict[key] is None:
                arg_dict[key] = value
            elif key in arg_dict and arg_dict[key] is not None:
                if isinstance(arg_dict[key], dict):
                    arg_dict[key] = update_nested_dict(value, arg_dict[key])
            else:
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
    return args


def atoi(string):
    """
    Convert a string digit to int if digit else return string
    :param string: a string
    :return: a string
    """
    return int(string) if string.isdigit() else string


def natural_keys(string):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '"""
    return [atoi(c) for c in re.split(r'(\d+)', string)]


def get_models_from_entry(list_file):
    """
    Get models weight (.pth) and associate settings files and read it from entry. Look in the same folder as settings files.
    So folder need to be separate for each settings file
    :param list_file: Can be a folder or settings yaml file
    :return: A dict with key segmentation model and value yaml file path
    """

    if len(list_file) == 1:
        if os.path.isdir(list_file[0]):
            all_yaml_files = [x for x in glob(f'{list_file[0]}/**/*.yaml', recursive=True)]
        else:
            all_yaml_files = list_file
    else:
        if not all(x.lower().endswith('.yaml') for x in list_file):
            raise ValueError("On all files a .yaml extension is needed")
        else:
            all_yaml_files = list_file
    all_yaml_files.sort(key=natural_keys)
    model_dict = {}

    # Now look in each folder of settings files
    for settings_files in all_yaml_files:
        folder_path = os.path.dirname(settings_files)
        models = [x for x in glob(f'{folder_path}/**/*.pth', recursive=True)]
        models.sort(key=natural_keys)
        if len(models) == 0:
            raise ValueError(f"No model.pth was found in {folder_path}. There is an ambiguity to"
                             f"associate the model to the setting ")

        logger.info(f'{len(models)} weight(s) has been found in the folder: {folder_path}')

        for nb_model, model in enumerate(reversed(models)):
            model_dict[open(settings_files)] = model

    return model_dict


def create_patient_dict_classification(input_path, modality, label_path=None, patient_col=None, label_col=None):
    """

    :param input_path: input path of nii files
    :param modality: get files with the specified keyword ['t1', 't1ce', 't1ce_ss', 'flair', 't2', 'ct', 'seg']
                     (str or list)
    :param label_path: path of
    :param patient_col: patient columns (list of a unique str or string)
    :param label_col: labels columns (can be multiple, list of str or string)
    :return: dict for input of DL
    """
    if isinstance(modality, str):
        modality = [modality]

    if label_path is not None and (patient_col is None or label_col is None):
        raise TypeError("With a label_path, patient_col and labels_cols need to be specified")

    if label_col is not None and isinstance(label_col, str):
        label_col = [label_col]

    if patient_col is not None and isinstance(patient_col, (list, tuple)):
        if len(patient_col) != 1:
            raise ValueError("patient_col need to be unique")
        patient_col = str(patient_col[0])
    elif patient_col is not None and not isinstance(patient_col, str):
        raise TypeError("patient_col need to be a string or list with unique value")

    check_modality = ['t1', 't1ce', 't1ce_ss', 'flair', 't2', 'ct', 'seg']
    if not all(x.lower() in check_modality for x in modality):
        raise ValueError('The input modality is not supported')

    if input_path is not None:
        if not os.path.exists(input_path):
            raise FileExistsError("Input path is not correct")
        input_files = glob(os.path.join(input_path, "**/*.nii.gz"), recursive=True)
    else:
        raise FileExistsError("Input path is not correct")

    logger.debug(f"Data input path: {input_path}")

    input_files.sort(key=lambda x: os.path.basename(x))

    label_df = pd.DataFrame
    labels_cols_input = []
    if label_path is not None:
        if not os.path.isfile(label_path):
            raise FileNotFoundError('Label file not exist')

        if label_path.lower().endswith('.csv'):
            label_df = pd.read_csv(label_path)
        elif label_path.lower().endswith(('.xlsx', 'xls')):
            label_df = pd.read_excel(label_path)
        else:
            raise FileExistsError('Label file must be excel or csv type')

        #  label file need to contain a column patient and label
        if patient_col is not None and label_col is not None:
            patient_col = [col for col in label_df.columns if patient_col in col]
            labels_cols_input = [col for col in label_df.columns for label in label_col if label in col]

            if len(patient_col) != 1:
                raise ValueError('Patient columns or labels columns is not unique in file or is Missing !')
            patient_col = str(patient_col[0])

            if len(labels_cols_input) != len(label_col):
                raise ValueError(
                    f'Found {labels_cols_input} in files, but the input needed was {label_col}, check your input')

    patient_dict = OrderedDict()
    for file in input_files:
        input_file = {}
        path, fname, ext = split_filename(file)
        patient_id = os.path.basename(os.path.normpath(path))
        if patient_id not in patient_dict:
            patient_dict[patient_id] = {}
        modality_from_file, label_from_file = [], []
        for idx, ele in enumerate(check_modality):
            if ele in file:
                if ele == 't1' and 't1ce' not in file:
                    modality_from_file = ele
                else:
                    modality_from_file = ele

        if modality_from_file:
            input_file[modality_from_file] = file
        else:
            continue
        patient_dict[patient_id].update(input_file)

    if None not in (label_path, patient_col, label_col):
        for patient_id in list(patient_dict.keys()):
            if not any(label_df[patient_col].str.contains(patient_id, flags=re.IGNORECASE).to_list()):
                # remove patient of the patient_dict that are not in the label file
                del patient_dict[patient_id]
                continue
            for lab in labels_cols_input:
                label_value = label_df[label_df[patient_col].str.contains(patient_id, flags=re.IGNORECASE)]
                if label_value.empty:  # patient from input file is not in label file so continue
                    continue
                label_value = label_value[lab].squeeze()
                if label_value != label_value:  # is NaN
                    raise ValueError(f'In label columns {lab}, the patient {patient_id} has a {label_value} value')
                elif not isinstance(label_value, (float, int, np.integer)):
                    raise TypeError(f'type of label {lab} for the patient {patient_id} is {type(label_value)}')

                patient_dict[patient_id].update({lab: label_value})

        # check all patient have same key with value not empty
        check_patient_dict(patient_dict, label_col + modality)

    return patient_dict


def check_patient_dict(patient_dict, keys):
    """
    check all keys and associated value are present in the patient dict
    :param patient_dict: patient_dict
    :param keys: keys to check for a patient
    """
    if not isinstance(keys, (tuple, list)):
        keys = [keys]
    for patient_id in patient_dict:
        if not all(k in patient_dict[patient_id] for k in keys):
            raise ValueError(f'A modality or label is missing for {patient_id}: {patient_dict[patient_id]}')
        for key in keys:
            if patient_dict[patient_id][key] in ['', [], ()]:  # value is missing
                raise ValueError(f'Empty value {patient_dict[patient_id][key]} for {key} for patient {patient_id}')


def create_dataset_files(patient_dict, patient_ids, modality, labels_name=None):
    """
    From the patient dict create the train and val dataset files for input.
    :param patient_dict: patient dict
    :param patient_ids: patient ID
    :param modality: choice of modality
    :param labels_name: label_col corresponds to the classes
    :return data files
    """
    if isinstance(modality, str):
        modality = [modality]

    if labels_name is not None and isinstance(labels_name, str):
        labels_name = [labels_name]

    modality_copy = copy.deepcopy(modality)
    if 'seg' in modality_copy:
        modality_copy.remove('seg')

    data_files = []
    for patient, patient_images in patient_dict.items():
        files, val = {}, {}
        if patient in patient_ids:
            if 'seg' in modality:  # do that to create transform based on seg
                files['seg'] = patient_images['seg']

            files['input'] = [patient_images[x] for x in modality_copy]

            if labels_name is not None:
                if len(labels_name) == 1:
                    files['label'] = patient_images[labels_name[0]]
                else:
                    files['label'] = [patient_images[x] for x in labels_name]

            files['patient_id'] = patient
            data_files.append(files)
    return data_files


def get_transforms_settings(transforms_settings):
    """
    Get transforms from the config file
    :param transforms_settings: Monai transform settings
    Returns Transform list
    """
    transform_list = []
    for transform, args_transform in transforms_settings.items():
        transform_list.append(getattr(transforms, transform)(**args_transform))
    return transform_list


def get_transforms(train_transforms, val_transforms):
    """
    Get transforms for train and val
    :param train_transforms: train transforms
    :param val_transforms: val transforms
    Returns Compose of transform for train, val
    """
    train_transforms = get_transforms_settings(train_transforms)
    val_transforms = get_transforms_settings(val_transforms)
    return Compose(train_transforms), Compose(val_transforms)


def get_model(model_type, model_params):
    """
    Get model
    :param model_type: model type
    :param model_params: model params
    :return: model function
    """
    if hasattr(networks, model_type):  # Check on models
        model = getattr(networks, model_type)
    else:
        raise ValueError('The Model is not correct')

    if model_params not in [None, 'None']:
        model = model(**model_params)
    else:
        model = model()

    return model


def shape_to_model(data, min_shape=None):
    """
    From a tensor with random size, return a feasible shape for the model
    :param data: tensor
    :param: min_shape shape for input to the model
    :returns tensor shape to model, p_b, p_a
    """
    assert torch.is_tensor(data), "Support tensor to speed computation time"

    # consider channel first
    if data.ndim == 5:
        datashape = data.shape[2:]
    elif data.ndim == 4:
        datashape = data.shape[1:]
    else:
        raise ValueError("Tensor dimension is incorrect")

    zeropad_shape = np.ceil(np.divide(datashape, 16)).astype(int) * 16
    if min_shape is not None:
        zeropad_shape[zeropad_shape < min_shape] = min_shape

    p = zeropad_shape - datashape  # padding
    p_b = np.ceil(p / 2).astype(int)  # padding before image
    p_a = np.floor(p / 2).astype(int)  # padding after image
    data_pad = F.pad(data, (p_b[2], p_a[2], p_b[1], p_a[1], p_b[0], p_a[0]), mode='constant', value=0)

    return data_pad, p_b, p_a


def apply_activation(tensor, sigmoid=False, softmax=False):
    if sigmoid and softmax:
        raise ValueError("sigmoid=True and softmax=True are not compatible.")

    if sigmoid:
        tensor = torch.sigmoid(tensor)
    elif softmax:
        tensor = torch.softmax(tensor, 1)
    else:
        pass

    return tensor


def do_pred(x, models, sigmoid=False, softmax=False):
    if not isinstance(models, (tuple, list)):
        models = [models]

    outputs_models = []
    for model in models:
        outputs_tta = []
        out = apply_activation(model(x), sigmoid, softmax)
        outputs_tta.append(out.cpu())
            
        outputs_models.extend(outputs_tta)
    outputs = torch.stack(outputs_models).mean(dim=0)

    return outputs
