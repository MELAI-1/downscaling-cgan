import os
from pathlib import Path
import hashlib
import json
import yaml
import re
import copy
from glob import glob
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from calendar import monthrange
from typing import Iterable, Tuple, Callable, List
from timezonefinder import TimezoneFinder
from dateutil import tz
from types import SimpleNamespace

tz_finder = TimezoneFinder()
from_zone = tz.gettz('UTC')

##🚩update the special areas 

special_areas = {'all': {'lat_range': None, 'abbrv': 'ALL'},
                 'lake_victoria': {'lat_range': [-4.05,2.05], 'lon_range': [29.05, 37.05], 'abbrv': 'LV'},
                 'somalia': {'lat_range': [-1.05,4.05], 'lon_range': [40.0, 44.05],  'abbrv': 'S'},
                 'coast': {'lat_range': [-11.05, -4.70 ], 'lon_range': [38.0,39.0],  'abbrv': 'C'},
                 'west_lv_basin': {'lat_range': [-4.70,0.30], 'lon_range': [29.5,31.3],  'abbrv': 'WLVB'},
                 'east_lv_basin': {'lat_range': [-3.15, 1.55], 'lon_range': [34.5,36.0],  'abbrv': 'ELVB'},
                 'nw_ethiopian_highlands': {'lat_range': [6.10, 14.15], 'lon_range': [34.60, 40.30], 'abbrv': 'NWEH'},
                 'kenya': {'lat_range': [-4.65, 5.15], 'lon_range': [34.05, 42.15], 'abbrv': 'K'},
                 'kenya_core': {'lat_range': [-4.05, 1.15], 'lon_range': [35.05, 39.05], 'abbrv': 'KC'}

}

def hash_dict(params: dict):
    h = hashlib.shake_256()
    h.update(json.dumps(params, sort_keys=True).encode('utf-8'))
    return h.hexdigest(8)


def load_yaml_file(fpath: str):  
    with open(fpath, 'r') as f:
        data = yaml.safe_load(f)
        
    return data

def write_to_yaml(fpath: str, data: dict):
    
    with open(fpath, 'w+') as ofh:
        yaml.dump(data, ofh, default_flow_style=False)
        
def date_range_from_year_month_range(year_month_ranges):
    
    if not isinstance(year_month_ranges[0], list):
        year_month_ranges = [year_month_ranges]
    
    output_dates = []
    for ym_range in year_month_ranges:
        start_date = datetime.datetime(year=int(ym_range[0][:4]), month=int(ym_range[0][4:6]), day=1)
        
        end_year = int(ym_range[-1][:4])
        end_month = int(ym_range[-1][4:6])
        end_date = datetime.datetime(year=end_year, 
                                    month=end_month, 
                                    day=monthrange(end_year, end_month)[1])
        output_dates += [item.date() for item in pd.date_range(start=start_date, end=end_date)]
    return sorted(set(output_dates))

def get_checkpoint_model_numbers(log_folder: str) -> List[int]:
    """Get list of iteration points that have been saved as checkpoints

    Args:
        log_folder (str): Base folder; must contain a 'models' directory
        
    Returns:
        list[int]: list of interger model numbers
    """
    models_glob = str(Path(log_folder).parents[0] / 'models/*.h5')

    fps = glob(models_glob)
    
    if fps:
        model_numbers = [int(re.search(r'([0-9]+).h5', fp).groups()[0]) for fp in fps]
    else:
        return []
    
    return model_numbers
    

def get_best_model_number(log_folder: str):
    """
    Get model number that has lowest value according to metric specified (defaults to CRPS)

    Args:
        log_folder (str): Path to evaluation results
        metric_column_name (str): Name of column to optimise on. Defaults to CRPS_no_pooling 

    Returns:
        int: Model number
    """
    df = pd.read_csv(os.path.join(log_folder, 'eval_validation.csv'))

    # find model_number with lowest CRPS
    min_crps = df['CRPS_no_pooling'].min()
    model_number = df[df.CRPS_no_pooling == min_crps]['N'].values[0]
    
    return model_number


def get_valid_quantiles(data_size: int, min_data_points_per_quantile: int, raw_quantile_locations: list) -> list:
    """Returns a list of quantiles q such that (1-q)*input_data.size > min_data_points_per_quantile

    Args:
        data_size (int): Total number of data points
        min_data_points_per_quantile (int): Minimum number of data points per quantile
        raw_quantile_locations (list): The raw quantile locations, to be truncated

    Returns:
        list: quantile locations with necessary quantiles removed
    """

    quantile_locs = [val for val in raw_quantile_locations[:-1] if (1-val)*data_size > min_data_points_per_quantile]
    
    return quantile_locs

def resample_array(input_array: np.ndarray, time_resample: bool=False) -> np.ndarray:
    """Resample a single array with replacement

    Args:
        input_array (np.ndarray): array to resample
        time_resample (bool, optional): If True then will only resample along the 0 axis. Defaults to False.

    Returns:
        np.ndarray: resampled array
    """
    shape = input_array.shape
    if time_resample:

        idx = np.random.randint(0, shape[0], shape[0])

        return input_array[idx, ...]
    else:
        
        return np.random.choice(input_array.flatten(), size=shape, replace=True) 

    

def resample_paired_arrays(fcst_array: np.ndarray, 
                           obs_array: np.ndarray, time_resample: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function for resampling paired arrays (e.g. for bootstrapping)

    Args:
        fcst_array (np.ndarray): forecast array
        obs_array (np.ndarray): observation array
        time_resample (bool, optional): If True, will only resample along the 0 axis. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: resampled forecast and observation arrays
    """
    shape = fcst_array.shape
    
    if time_resample:
        
        idx = np.random.randint(0, shape[0], shape[0])

        resampled_fcst = fcst_array[idx, ...]
        resampled_obs = obs_array[idx, ...]
    else:
        idx = np.random.randint(0,fcst_array.size, shape)

        resampled_fcst = np.take(fcst_array, idx)
        resampled_obs = np.take(obs_array, idx)

    return resampled_fcst, resampled_obs

def bootstrap_summary_statistic(statistic_func: Callable[..., float],
                                input_array: np.ndarray,
                                n_bootstrap_samples: int,
                                time_resample: bool=False,
                                **kwargs):
    
    
    tmp_results = []
    for _ in tqdm(range(n_bootstrap_samples)):
        resampled_array = resample_array(input_array, time_resample)
        tmp_results.append(statistic_func(resampled_array, **kwargs))
        
    tmp_results = np.stack(tmp_results, axis=0)
    # calculate mean and standard deviation
    output_dict = {'mean': tmp_results.mean(axis=0), 'std': tmp_results.std(axis=0)}
    
    return output_dict

def subsample_summary_statistic(statistic_func: Callable[..., float],
                                input_array: np.ndarray,
                                n_subsamples: int,
                                axis: int=0,
                                **kwargs):
    
        
    split_arrays = np.array_split(input_array, n_subsamples, axis=axis)

    tmp_results = []

    for arr in tqdm(split_arrays):
        
        tmp_results.append(statistic_func(arr, **kwargs))
            
    tmp_results = np.stack(tmp_results, axis=0)
    
    # calculate mean and standard deviation
    output_dict = {'mean': tmp_results.mean(axis=0), 'std': tmp_results.std(axis=0)}
    
    return output_dict 

    
def bootstrap_metric_function(metric_func: Callable[..., float], 
                              fcst_array: np.ndarray, 
                              obs_array: np.ndarray, 
                              n_bootstrap_samples: int, 
                              time_resample: bool=False,
                              **kwargs) -> dict:
    """Wrapper to run bootstrap on metric function
    The function must accept a forecast array and observation array, plus keyword arguments

    Args:
        metric_func (Callable[..., float]): Metric function
        fcst_array (np.ndarray): Forecast array
        obs_array (np.ndarray): Observation array
        n_bootstrap_samples (int): Number of bootstrap samples to generate

    Returns:
        dict: Summary statistics of metric
    """
    tmp_results = []
    for _ in tqdm(range(n_bootstrap_samples)):
        resampled_fcst_array, resampled_obs_array = resample_paired_arrays(fcst_array, obs_array, time_resample)
        tmp_results.append(metric_func(resampled_obs_array, resampled_fcst_array, **kwargs))

    # calculate mean and standard deviation
    output_dict = {'mean': np.mean(tmp_results), 'std': np.std(tmp_results)}
    
    return output_dict


def get_local_datetime(utc_datetime: datetime.datetime, longitude: float, latitude: float) -> datetime.datetime:
    """Get datetime at locality defined by lat,long, from UTC datetime

    Args:
        utc_datetime (datetime.datetime): UTC datetime
        longitude (float): longitude
        latitude (float): latitude

    Returns:
        datetime.datetime: Datetime in local time
    """
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_datetime = utc_datetime.astimezone(to_zone)
    
    return local_datetime

def get_local_hour(hour: int, longitude: float, latitude: float):
    """Convert hour to hour in locality.
    
    This is not as precise as get_local_datetime, as it won't contain information about e.g. BST. 
    But it is useful when the date is not known but the hour is

    Args:
        hour (int): UTC hour
        longitude (float): longitude
        latitude (float): latitude
    Returns:
        int: approximate hour in local time
    """
    utc_datetime = datetime.datetime(year=2000, month=1, day=1, hour=hour)
    utc_datetime.replace(tzinfo=from_zone)
    
    timezone = tz_finder.timezone_at(lng=longitude, lat=latitude)
    to_zone = tz.gettz(timezone)

    local_hour = utc_datetime.astimezone(to_zone).hour
    
    return local_hour

def convert_namespace_to_dict(ns_obj: SimpleNamespace) -> dict:
    """Convert nested namespace object to dict

    Args:
        ns_obj (SimpleNamespace): nested namespace

    Returns:
        dict: namespace converted into dict
    """
    output_dict = copy.deepcopy(ns_obj).__dict__
    for k, v in output_dict.items():
        if isinstance(v, SimpleNamespace):
            output_dict[k] = v.__dict__
            
    return output_dict

def get_area_range(data_config, area, special_areas=special_areas):

    special_areas = copy.deepcopy(special_areas)

    latitude_range=np.arange( np.round(data_config.min_latitude,2), np.round(data_config.max_latitude,2), data_config.latitude_step_size)
    longitude_range=np.arange( np.round(data_config.min_longitude,2),  np.round(data_config.max_longitude,2), data_config.longitude_step_size)

    lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
    lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

    special_areas['all']['lat_range'] = [np.round(min(latitude_range),2), np.round(max(latitude_range),2)]
    special_areas['all']['lon_range'] =  [np.round(min(longitude_range),2), np.round(max(longitude_range),2)]

    for k, v in special_areas.items():
        lat_vals = [lt for lt in lat_range_list if np.round(v['lat_range'][0],2) <=  np.round(lt,2) <= np.round(v['lat_range'][1],2)]
        lon_vals = [ln for ln in lon_range_list if np.round(v['lon_range'][0],2) <= np.round(ln,2) <= np.round(v['lon_range'][1],2)]
        if lat_vals and lon_vals:
    
            special_areas[k]['lat_index_range'] = [lat_range_list.index(lat_vals[0]), lat_range_list.index(lat_vals[-1])]
            special_areas[k]['lon_index_range'] = [lon_range_list.index(lon_vals[0]), lon_range_list.index(lon_vals[-1])]

    lat_range_index = special_areas[area]['lat_index_range']
    lon_range_index = special_areas[area]['lon_index_range']
    latitude_range=np.arange(special_areas[area]['lat_range'][0], special_areas[area]['lat_range'][-1] + data_config.latitude_step_size, data_config.latitude_step_size)
    longitude_range=np.arange(special_areas[area]['lon_range'][0], special_areas[area]['lon_range'][-1] + data_config.longitude_step_size, data_config.longitude_step_size)

    return latitude_range, lat_range_index, longitude_range, lon_range_index

