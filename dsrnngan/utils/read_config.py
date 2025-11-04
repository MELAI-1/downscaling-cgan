import os
import sys
import copy
import tensorflow as tf
import numpy as np
import types
import math
from pathlib import Path

HOME = Path(__file__).parents[2]
CONFIG_FOLDER = HOME / 'config'

from dsrnngan.utils.utils import load_yaml_file
from dsrnngan.model.wloss import CL_OPTIONS_DICT

def read_config(config_filename: str=None, config_folder: str=CONFIG_FOLDER) -> dict:

    config_path = os.path.join(config_folder, config_filename)
    try:
        config_dict = load_yaml_file(config_path)
    except FileNotFoundError as e:
        print(e)
        print(f"You must set {config_filename} in the config folder.")
        sys.exit(1)       
    
    return config_dict

def read_model_config(config_filename: str='model_config.yaml', config_folder: str=CONFIG_FOLDER,
                      model_config_dict: dict=None) -> dict:
    
    if model_config_dict is None:
        model_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
      
    model_config = copy.deepcopy(model_config_dict)
    for k, v in model_config.items():
        if isinstance(v, dict):
            model_config[k] = types.SimpleNamespace(**v)
    model_config = types.SimpleNamespace(**model_config)
    
    model_config.train.num_epochs = model_config_dict["train"].get("num_epochs")
    model_config.train.num_samples = model_config_dict['train'].get('num_samples') # leaving this in while we transition to using epochs    
    model_config.train.crop_size = model_config_dict['train'].get('img_chunk_width')
    model_config.train.kl_weight = float(model_config.train.kl_weight)
    model_config.train.content_loss_weight = float(model_config.train.content_loss_weight)
    model_config.train.ensemble_size = model_config_dict['train'].get('ensemble_size')
    model_config.train.training_ratio = model_config_dict['train'].get('training_ratio', 5)
    model_config.train.rotate = model_config_dict['train'].get('rotate', False)

    model_config.val.val_range = model_config_dict['val'].get('val_range')
    model_config.val.val_size = model_config_dict.get("val", {}).get("val_size")
    
    model_config.generator.learning_rate_gen = float(model_config.generator.learning_rate_gen)
    model_config.generator.output_activation = model_config_dict['generator'].get('output_activation', 'softplus')
    model_config.generator.normalisation =  model_config_dict['generator'].get('normalisation')
    
    model_config.discriminator.learning_rate_disc = float(model_config.discriminator.learning_rate_disc)
    model_config.discriminator.normalisation = model_config_dict['discriminator'].get('normalisation')
    
    if model_config.mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
    
    possible_cl_types = list(CL_OPTIONS_DICT.keys())
    if model_config.train.CL_type not in possible_cl_types:
        raise ValueError(f"Content loss type is restricted to {possible_cl_types}")
    
    if model_config.train.num_samples is None:
        if model_config.train.num_epochs is None:
            raise ValueError('Must specify either num_epochs or num_samples')
        
    assert math.prod(model_config.downscaling_steps) == model_config.downscaling_factor, "downscaling factor steps do not multiply to total downscaling factor!"

    return model_config

def read_data_config(config_filename: str='data_config.yaml', config_folder: str=CONFIG_FOLDER,
                     data_config_dict: dict=None) -> dict:
    if data_config_dict is None:
        
        data_config_dict = read_config(config_filename=config_filename, config_folder=config_folder)
    
    data_config_ns = types.SimpleNamespace(**data_config_dict)
    data_config_ns.load_constants = data_config_dict.get('load_constants', True)
    data_config_ns.input_channels = len(data_config_ns.input_fields)
    data_config_ns.class_bin_boundaries = data_config_dict.get('class_bin_boundaries')
    
    if data_config_ns.class_bin_boundaries is not None:
        if 0 not in data_config_ns.class_bin_boundaries:
            data_config_ns.class_bin_boundaries = [0] + data_config_ns.class_bin_boundaries
            
        if len(data_config_ns.class_bin_boundaries) != data_config_ns.num_classes:
            raise ValueError('Class bin boundary lenght not consistent with number of classes')
    
    # For backwards compatability
    if isinstance(data_config_ns.constant_fields, int):
        data_config_ns.constant_fields = ['orography', 'lsm']
    
    data_config_ns.normalise_inputs = data_config_dict.get('normalise_inputs', False)
    
    if data_config_dict.get('normalise_outputs') is False:
            data_config_ns.output_normalisation = None
    else:
        data_config_ns.output_normalisation = data_config_dict.get('output_normalisation', "log")
        
    # # Infer input image dimensions if not given
    # latitude_range, longitude_range = get_lat_lon_range_from_config(data_config=data_config_ns)
    # data_config_ns.input_image_height = len(latitude_range)
    # data_config_ns.input_image_width = len(longitude_range)
    data_config_ns.input_image_height = 384
    data_config_ns.input_image_width  = 352
    # data_config_ns.input_image_channels = 11
    # data_config_ns.constants_image_channels = 2
    # data_config_ns.output_image_channels = 1


    return data_config_ns

def get_data_paths(config_folder: str=CONFIG_FOLDER, data_config: types.SimpleNamespace=None):
    
    if data_config is None:
        data_config = read_data_config(config_folder=config_folder)
        
    if isinstance(data_config, dict):
        data_config = types.SimpleNamespace(**copy.deepcopy(data_config))
        
    data_paths = data_config.paths[data_config.data_paths]
    return data_paths


def set_gpu_mode(model_config: dict=None):
    
    if model_config is None:
        model_config = read_model_config()
        
    if model_config.use_gpu:
        print('Setting up GPU', flush=True)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # remove environment variable (if it doesn't exist, nothing happens)
        
        # set memory allocation to incremental if desired
        # I think this is to ensure the GPU doesn't run out of memory
        if model_config.gpu_mem_incr:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            # if gpu_mem_incr False, do nothing
            pass
        if model_config.disable_tf32:
            tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        # if use_gpu False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
def get_lat_lon_range_from_config(data_config=None):
    
    if data_config is None:
        data_config = read_data_config()
    
    min_latitude = data_config.min_latitude
    max_latitude = data_config.max_latitude
    latitude_step_size = data_config.latitude_step_size
    min_longitude = data_config.min_longitude
    max_longitude = data_config.max_longitude
    longitude_step_size = data_config.longitude_step_size
    
    latitude_range=np.arange(min_latitude, max_latitude, latitude_step_size)
    longitude_range=np.arange(min_longitude, max_longitude, longitude_step_size)
    
    return latitude_range, longitude_range


def get_config_objects(config: dict):

    # Convert old style config into namespace objects; soon to be replaced 
    # once legacy model runs are no longer in use  
    model_config = config['MODEL']
    data_config = config['DATA']

    model_config['train'] = config['TRAIN']
    model_config['generator'] = config['GENERATOR']
    model_config['discriminator'] = config['DISCRIMINATOR']
    model_config['val'] = config['VAL']


    for k, lp in config['LOCAL'].items():
        model_config[k] = lp
    
    data_config['data_paths'] = config['LOCAL']['data_paths']
    
    for k, dp in config['DOWNSCALING'].items():
        model_config[k] = dp
    
    model_config['downscaling_steps'] = model_config['steps']
        
    model_config = read_model_config(model_config_dict=model_config)
    data_config = read_data_config(data_config_dict=data_config)
        
    return model_config, data_config

    def update_data_config_for_bucket(data_config: types.SimpleNamespace, bucket_url: str):
        """
        Update the data_config object to use an online bucket for data paths.

        Args:
            data_config (types.SimpleNamespace): The data config object.
            bucket_url (str): The URL or path to the online bucket.

        Returns:
            types.SimpleNamespace: Updated data config object.
        """
        # Assume data_paths is a dictionary of keys to local paths
        # Replace each path with the bucket URL + relative path
        if hasattr(data_config, "paths") and isinstance(data_config.paths, dict):
            updated_paths = {}
            for key, path in data_config.paths.items():
                # If path is already a URL, skip; otherwise, prepend bucket_url
                if isinstance(path, str) and not path.startswith("http"):
                    updated_paths[key] = f"{bucket_url.rstrip('/')}/{path.lstrip('/')}"
                else:
                    updated_paths[key] = path
            data_config.paths = updated_paths
        return data_config