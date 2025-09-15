""" Data generator class for full-image evaluation of precipitation downscaling network """
import random
import re
import numpy as np
import tensorflow as tf
from types import SimpleNamespace
from typing import Union, Iterable
from tensorflow.keras.utils import Sequence

all_ngcm_hours = [0, 6, 12, 18] ##🚩ngcm hours

from dsrnngan.data.data import load_fcst_radar_batch, load_hires_constants, all_fcst_hours, DATA_PATHS, all_ifs_fields, all_era5_fields, input_fields
##🚩import ngcm function 
from dsrnngan.data.data import load_ngcm, all_ngcm_fields,  get_ngcm_stats, all_ngcm_fields
from dsrnngan.utils.read_config import read_model_config, get_data_paths, get_lat_lon_range_from_config


##🚩fields lookup
fields_lookup = {'ifs': all_ifs_fields, 'era5': all_era5_fields, 'ngcm': all_ngcm_fields}

class DataGenerator(Sequence):
    def __init__(self, dates: list, batch_size: int, data_config: SimpleNamespace,
                 shuffle: bool=True, hour: Union[int, str, list, np.ndarray]='random',
                 downsample: bool=False, repeat_data: bool=False, seed: int=None):
        
        if seed is not None:
            random.seed(seed)
        self.dates = dates
        self.forecast_data_source = data_config.fcst_data_source
        self.observational_data_source = data_config.obs_data_source
        self.data_paths = get_data_paths(data_config=data_config)

        if isinstance(hour, str):
            if hour == 'random':
                self.hours = np.repeat(all_fcst_hours, len(self.dates))
                self.dates = np.tile(self.dates, len(all_fcst_hours))
            else:
                assert False, f"Unsupported hour {hour}"

        elif isinstance(hour, (int, np.integer)):
            self.hours = np.repeat(hour, len(self.dates))
            self.dates = np.tile(self.dates, 1)  # lol

        elif isinstance(hour, (list, np.ndarray)):
            if len(hour) == len(self.dates):
                # In this case we assume that the user is passing in pairs of (date,hour) combinations
                self.hours = np.array(hour)
            else:
                self.hours = np.repeat(hour, len(self.dates))
                self.dates = np.tile(self.dates, len(hour))

            if self.forecast_data_source == 'era5':
                raise ValueError('ERA5 data only supports daily')
            #  ##🚩update this part if self.forecast_data_source == 'ngcm': 
        else:
            raise ValueError(f"Unsupported hour {hour}")
    
        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)
            self.shuffle_data()

        self.batch_size = batch_size
        self.repeat_data = repeat_data

        self.fcst_fields = data_config.input_fields
        self.constant_fields = data_config.constant_fields
        self.shuffle = shuffle
        self.hour = hour
        self.normalise_inputs = data_config.normalise_inputs
        self.normalisation_strategy = {field: data_config.ngcm_input_normalisation_strategy[re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', field)] for field in self.fcst_fields}
        self.output_normalisation = data_config.output_normalisation

        self.downsample = downsample
        self.latitude_range, self.longitude_range = get_lat_lon_range_from_config(data_config)
               
        if self.downsample:
            # read downscaling factor from file
            self.ds_factor = read_model_config().downscaling_factor
        
        if self.constant_fields is None:
            # Dummy constants
            self.constants = np.ones((self.batch_size, len(self.latitude_range), len(self.longitude_range), 1))
        else:
            self.constants = load_hires_constants(
                                                  fields=self.constant_fields,
                                                  data_paths=self.data_paths['GENERAL'],
                                                  batch_size=self.batch_size,
                                                  latitude_vals=self.latitude_range, longitude_vals=self.longitude_range)
    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_downsampler(self, radar):
        kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
        image = tf.nn.conv2d(radar, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
                             name='conv_debug', data_format='NHWC')
        return image

    def __getitem__(self, idx):
        
        if self.repeat_data:
            # Repeat data indexes when it gets past the maximum
            idx = idx % len(self.dates)
            
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]
        hours_batch = self.hours[idx*self.batch_size:(idx+1)*self.batch_size]
        
        # Load and return this batch of images
        data_x_batch, data_y_batch = load_fcst_radar_batch(
            dates_batch,
            fcst_dir=self.data_paths['GENERAL'][self.forecast_data_source.upper()],
            obs_data_dir=self.data_paths['GENERAL'][self.observational_data_source.upper()],
            constants_dir=self.data_paths['GENERAL']['CONSTANTS'],
            constant_fields=None, # These are already loaded separately
            fcst_fields=self.fcst_fields,
            fcst_data_source=self.forecast_data_source,
            obs_data_source=self.observational_data_source,
            hour=hours_batch,
            normalise_inputs=self.normalise_inputs,
            normalisation_strategy=self.normalisation_strategy,
            output_normalisation=self.output_normalisation,
            latitude_range=self.latitude_range,
            longitude_range=self.longitude_range)
        print(f"Loaded data_x_batch shape: {data_x_batch.shape}, data_y_batch shape: {data_y_batch.shape}")
        
        if self.downsample:
            # replace forecast data by coarsened radar data!
            data_x_batch = self._dataset_downsampler(data_y_batch[..., np.newaxis])

        return {"lo_res_inputs": data_x_batch,
                "hi_res_inputs": self.constants,
                "dates": dates_batch, "hours": hours_batch},\
                {"output": data_y_batch}


    def shuffle_data(self):
        assert len(self.hours) == len(self.dates)
        p = np.random.permutation(len(self.hours))
        self.hours = self.hours[p]
        self.dates = self.dates[p]
        return

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()

class PermutedDataGenerator(Sequence):
    """
    A class designed to mimic the data generator, but returning values where the inputs have been permuted
    
    Designed to be used in a situation where data has already been gathered, but we want to do things like permuted feature importance
    """
    def __init__(self, lo_res_inputs: np.ndarray, hi_res_inputs: np.ndarray, outputs: np.ndarray, dates: np.ndarray, hours: np.ndarray,
                 input_permutation_config: dict, seed: int=None):
            
            self.lo_res_inputs = lo_res_inputs
            self.hi_res_inputs = hi_res_inputs
            self.outputs = outputs
            self.dates = dates
            self.hours = hours
     
            self.permutation_type = input_permutation_config['type']
            self.permute_index = input_permutation_config['permute_index']
            # Shuffle inputs for peremutation feature importance
            if self.permutation_type == 'lo_res_inputs':
                num_shufflable_vars = self.lo_res_inputs.shape[-1]
            elif self.permutation_type == 'hi_res_inputs':
                num_shufflable_vars = self.lo_res_inputs.shape[-1]
            else:
                raise ValueError(f'Unrecognised shuffle_type: {self.permutation_type}')
            
            if self.permute_index > num_shufflable_vars:
                raise ValueError(f'shuffle_index for {self.permutation_type} must be less than {num_shufflable_vars} for this data')
                
            # Create permuted indexes       
            random.seed(seed)
            self.permuted_indexes = random.sample(list(range(len(self.dates))), len(self.dates))

    def __getitem__(self, idx: int):
        
        lo_res_inputs = self.lo_res_inputs.copy()
        hi_res_inputs = self.hi_res_inputs.copy()
        
        if self.permutation_type == 'lo_res_inputs':

            lo_res_inputs[:,:,:,self.permute_index] = lo_res_inputs[self.permuted_indexes, :, :, self.permute_index]

        elif self.permutation_type == 'hi_res_inputs':

            hi_res_inputs[:,:,:,self.permute_index] = hi_res_inputs[self.permuted_indexes, :, :, self.permute_index]
        
        return {"lo_res_inputs": lo_res_inputs[idx:idx+1, :, :, :],
                "hi_res_inputs": hi_res_inputs[idx:idx+1, :, :, :],
                "dates": self.dates[idx:idx+1], "hours": self.hours[idx:idx+1]},\
                {"output": self.outputs[idx:idx+1,:,:]}
                
if __name__ == "__main__":
    pass
