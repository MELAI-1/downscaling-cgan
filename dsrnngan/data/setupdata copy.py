import gc
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Iterable, Generator
from types import SimpleNamespace

from dsrnngan.data import tfrecords_generator, setupdata
from dsrnngan.data.tfrecords_generator import DataGenerator
from dsrnngan.data.data import DATA_PATHS
from dsrnngan.utils.utils import date_range_from_year_month_range, load_yaml_file
from dsrnngan.utils import read_config

HOME = Path(os.getcwd()).parents[1]


def setup_batch_gen(records_folder: str,
                    fcst_shape: tuple,
                    con_shape: tuple,
                    out_shape: tuple,
                    batch_size: int=64,
                    downsample: bool=False,
                    weights=None,
                    crop_size: int=None,
                    rotate: bool=False,
                    seed: int=None,
                    val=False,
                    val_fixed=False,
                    ):

    tfrecords_generator.return_dic = False
    print(f"downsample flag is {downsample}")
    train = DataGenerator('train',
                           batch_size=batch_size,
                           fcst_shape=fcst_shape,
                           con_shape=con_shape,
                           out_shape=out_shape,
                           downsample=downsample, 
                           weights=weights, 
                           crop_size=crop_size,
                           rotate=rotate,
                           records_folder=records_folder,
                           seed=seed)

    # note -- using create_fixed_dataset with a batch size not divisible by 16 will cause problems [is this true?]
    # create_fixed_dataset will not take a list
    
    # if val:
    #     # assume that val_size is small enough that we can just use one batch
    #     val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=val_size, downsample=downsample,
    #                                                    folder=records_folder)
    #     val = val.take(1)
    #     if val_fixed:
    #         val = val.cache()
    # else:
    #     val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=batch_size, downsample=downsample,
    #                                                    folder=records_folder)
    return train, None


def setup_full_image_dataset(
                             data_config,
                             year_month_ranges,
                             batch_size=2,
                             downsample=False,
                             hour='random',
                             shuffle=True,
                             ):

    from dsrnngan.data.data_generator import DataGenerator as DataGeneratorFull
    from dsrnngan.data.data import get_obs_dates

    date_range = date_range_from_year_month_range(year_month_ranges)
    # dates = get_obs_dates(date_range,
    #                       obs_data_source=data_config.obs_data_source, 
    #                       data_paths=read_config.get_data_paths(data_config=data_config),
    #                       hour=hour)

    data_full = DataGeneratorFull(dates=date_range,
                                  data_config=data_config,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  downsample=downsample,
                                  hour=hour)
    return data_full

def setup_data(data_config: SimpleNamespace,
               model_config: SimpleNamespace,
               fcst_shape: tuple[int]=None,
               con_shape: tuple[int]=None,
               out_shape: tuple[int]=None,
               records_folder: str=None,
               hour: str='random',
               weights: Iterable=None,
               load_full_image: bool=False,
               seed: int=None,
               shuffle: bool=True,
               full_image_batch_size: int=1) -> tuple[Generator]:
    """Setup data for training or validation

    Args:
        data_config (SimpleNamespace): data config object
        model_config (SimpleNamespace): model config object
        fcst_shape (tuple[int], optional): Shape of forecast data. 
        con_shape (tuple[int], optional): Shape of constant data. 
        out_shape (tuple[int], optional): Shape of output. 
        records_folder (str, optional): Folder with tfrecords in it (required if load_full_image=False). Defaults to None.
        hour (str, optional): Hour to load. Defaults to 'random'.
        weights (Iterable, optional): _description_. Defaults to None.
        load_full_image (bool, optional): If True, will load full images (e.g. for validation). Defaults to False.
        seed (int, optional): Seed to control shuffling in data generator. Defaults to None.
        shuffle (bool, optional): If true then data generators will shuffle data. Defaults to True.
        full_image_batch_size (int, optional): Batch size to use. Defaults to 1.

    Returns:
        tuple[Generator]: training data generator and validation data generator
    """

    if load_full_image:
        if model_config.train.training_range is None:
            batch_gen_train = None
        else:
            batch_gen_train = setup_full_image_dataset(data_config=data_config,
                                          year_month_ranges=model_config.train.training_range,
                                          batch_size=full_image_batch_size,
                                          downsample=model_config.downsample,
                                          hour=hour,
                                          shuffle=shuffle)
        if model_config.val.val_range is None:
            batch_gen_valid = None
        else:
            batch_gen_valid = setup_full_image_dataset(data_config=data_config,
                                          year_month_ranges=model_config.val.val_range,
                                          batch_size=full_image_batch_size,
                                          downsample=model_config.downsample,
                                          hour=hour,
                                          shuffle=shuffle)

    else:
        if not records_folder:
            raise ValueError('No records folder given')
        batch_gen_train, batch_gen_valid = setup_batch_gen(
            val=False,
            records_folder=records_folder,
            fcst_shape=fcst_shape,
            con_shape=con_shape,
            out_shape=out_shape,
            batch_size=model_config.train.batch_size,
            downsample=model_config.downsample,
            weights=weights,
            crop_size=model_config.train.crop_size,
            rotate=model_config.train.rotate,
            seed=seed)

    gc.collect()
    return batch_gen_train, batch_gen_valid



def load_data_from_config(config: dict, records_folder: str=None,
                          batch_size: int=1, load_full_image: bool=True, 
                          data_paths: dict=DATA_PATHS, hour: int='random',
                          fcst_fields: list=None):
    """Load data based on config file

    Args:
        config (dict): _description_
        records_folder (str, optional): Folder path to tf records. Defaults to None.
        batch_size (int, optional): Size of batches. Defaults to 1.
        load_full_image (bool, optional): If True will load full sized image. Defaults to True.
        data_paths (dict, optional): Dict containing data paths. Defaults to DATA_PATHS.
        hour (int, optional): Hour to load. Defaults to 'random'.

    Returns:
        _type_: _description_
    """
    
    model_config, _, _, data_config, _, _, train_config, _ = read_config.get_config_objects(config)
    
    data_gen_train, data_gen_valid = setupdata.setup_data(
        records_folder=records_folder,
        data_config=data_config,
        model_config=model_config,
        load_full_image=load_full_image,
        training_range=train_config.training_range,
        data_paths=data_paths,
        hour=hour)
    
    return data_gen_train, data_gen_valid

def load_data_from_folder(records_folder: str=None, 
                          batch_size: int=1, load_full_image: bool=True, 
                          data_paths: dict=DATA_PATHS):
    """
    Loads data from folder: if load full image is true then this fetches data directly from source,
    if not it will load from the previously created tfrecords

    Args:
        records_folder (str, optional): Folder with TF records in it; only required if load_full_image=False. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 1.
        load_full_image (bool, optional): Whether or not to load the full image. Defaults to True.
        data_paths (dict, optional): Dict of data paths
        

    Returns:
        data_gen_train, data_gen_valid: generators for training and validation data
    """
    setup_params = load_yaml_file(os.path.join(records_folder, 'local_config.yaml'))
    data_gen_train, data_gen_valid = load_data_from_config(setup_params, batch_size, load_full_image, data_paths)
    return data_gen_train, data_gen_valid



if __name__=='__main__':
    
    from dsrnngan.data.data import DATA_PATHS, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE, input_fields
    from dsrnngan.data.data import get_obs_dates
    
    parser = ArgumentParser(description='Gather training data for quantile mapping.')
    parser.add_argument('--n_samples', type=int, 
                        help='Number of samples to gather overall. -1 for all samples', default=-1)
    parser.add_argument('--output-dir', type=str, help='output directory',
                        )
    parser.add_argument('--log-folder', type=str,
                        help='Path to data that has already been generated by GAN, to get dates from.')
    parser.add_argument('--model-number', type=str,
                        help='Model number to use.')    
    parser.add_argument('--ym-index', type=int, default=None,
                       help="Index of which year month to load.")
    
    # parser.add_argument('-ym', '--ym-ranges', type=str, nargs='+', action='append', default=None,
    #                     help='Year month ranges to extract data from')
    args = parser.parse_args()
    
    with open(os.path.join(args.log_folder, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
        arrays = pickle.load(ifh)

    model_config = read_config.read_model_config(config_folder=args.model_folder)
    data_config = read_config.read_data_config(config_folder=args.model_folder)
    
    
    date_range = date_range_from_year_month_range(model_config.train.training_range)
    # all_dates = get_obs_dates(date_range, 
    #                     obs_data_source=obs_data_source, data_paths=DATA_PATHS)
    all_year_months = sorted(set([f"{d.year}{d.month:02d}" for d in date_range]))
    
    print(f'Total number of year months = {len(all_year_months)}')
    
    if args.ym_index is not None:
        all_year_months = [all_year_months[args.ym_index]]

    for ym in all_year_months:
        ym_range = [ym]

        if args.n_samples == -1:
            ym_date_range = date_range_from_year_month_range(ym_range)
            # ym_dates = get_obs_dates(ym_date_range[0], ym_date_range[-1], 
            #                     obs_data_source=obs_data_source, data_paths=DATA_PATHS)
            args.n_samples = 24*len(ym_date_range)
            hours = np.arange(0,24,1)
        else:
            n_hours = int(args.n_samples/len(ym_date_range))
            hours = np.random.choice(range(0,23), size=n_hours, replace=False)
            
        data_gen = setup_full_image_dataset(
                             data_config=data_config,
                             year_month_ranges=ym_range,
                             batch_size=2,
                             downsample=False,
                             hour=hours,
                             shuffle=False
                             )

        
        tpidx = input_fields.index('tp')
        
        obs_vals, fcst_vals, dates, hours = [], [], [], []
        
        data_idx = 0
            
        for kk in tqdm(range(args.n_samples)):
            
            try:
                inputs, outputs = data_gen[data_idx]
            except FileNotFoundError:
                print('Could not load file, attempting retries')
                success = False
                for retry in range(5):
                    data_idx += 1
                    print(f'Attempting retry {retry} of 5')
                    try:
                        inputs, outputs = data_gen[data_idx]
                        success = True
                    except FileNotFoundError:
                        pass
                if not success:
                    print(f'Stopping at {dates[data_idx]}')
                    raise FileNotFoundError
            
            inputs, outputs = data_gen[data_idx]

            try:
                cond = inputs['lo_res_inputs']
                fcst = cond[0, :, :, tpidx]
                const = inputs['hi_res_inputs']
                obs = outputs['output'][0, :, :]
                date = inputs['dates']
                hour = inputs['hours']
            except IndexError:
                print(inputs['dates'])
                print(data_idx)
                print(len(dates))
                
                break
            
            obs_vals.append(obs)
            fcst_vals.append(fcst)
            dates.append(date)
            hours.append(hour)
            
            data_idx += 1
            
        obs_array = np.stack(obs_vals, axis=0)
        fcst_array = np.stack(fcst_vals, axis=0)
            
        arrays = {'obs': obs_array, 'fcst_array': fcst_array, 
                'dates': dates, 'hours': hours}
        
        with open(os.path.join(args.output_dir, f"training_data_{'_'.join(ym_range)}_{args.n_samples}.pkl"), 'wb+') as ofh:
            pickle.dump(arrays, ofh, pickle.HIGHEST_PROTOCOL)