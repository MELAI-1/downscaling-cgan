import os
import glob
import random
import time
import numpy as np
import tensorflow as tf
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import git

from dsrnngan.utils import read_config, utils
from dsrnngan.data.data import file_exists, denormalise
from dsrnngan.utils.utils import hash_dict, write_to_yaml, date_range_from_year_month_range
from dsrnngan.utils.read_config import get_data_paths

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

return_dic = True

DATA_PATHS = read_config.get_data_paths()
records_folder = DATA_PATHS["TFRecords"]["tfrecords_path"]

#TODO: pass this as function argument instead
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

# Use autotune to tune the prefetching of records in parrallel to processing to improve performance
AUTOTUNE = tf.data.AUTOTUNE



## for the batch size I need to use 128 for the generation

def DataGenerator(data_label, batch_size, fcst_shape, con_shape, 
                  out_shape, repeat=True, 
                  downsample=False, weights=None, crop_size=None, rotate=False,
                  records_folder=records_folder, seed=None):
    return create_mixed_dataset(data_label, 
                                batch_size,
                                fcst_shape,
                                con_shape,
                                out_shape,
                                repeat=repeat, 
                                downsample=downsample, 
                                weights=weights, 
                                crop_size=crop_size,
                                rotate=rotate,
                                folder=records_folder, 
                                seed=seed)


def create_mixed_dataset(data_label: str,
                         batch_size: int,
                         fcst_shape: tuple[int, int, int],
                         con_shape: tuple[int, int, int],
                         out_shape: tuple[int, int, int],
                         repeat: bool=True,
                         downsample: bool=False,
                         folder: str=records_folder,
                         shuffle_size: int=256,
                         weights: list=None,
                         crop_size: int=None,
                         rotate: bool=False,
                         seed: int=None):
    """_summary_

    Args:
        year (int): year of data
        batch_size (int): size of batches
        fcst_shape (tuple[int, int, int], optional): shape of forecast input. Defaults to (20, 20, 9).
        con_shape (tuple[int, int, int], optional): shape of constants input. Defaults to (200, 200, 2).
        out_shape (tuple[int, int, int], optional): shape of output. Defaults to (200, 200, 1).
        repeat (bool, optional): repeat dataset or not. Defaults to True.
        downsample (bool, optional): whether to downsample or not. Defaults to False.
        folder (str, optional): folder containing tf records. Defaults to records_folder.
        shuffle_size (int, optional): buffer size of shuffling. Defaults to 256.
        weights (list, optional): list of floats, weights of classes when sampling. Defaults to None.
        crop_size (int, optional): Size to crop randomly crop images to.
        rotate (bool, optional): If true then data is agumented by random rotation.
        seed (int, optional): seed for shuffling and sampling. Defaults to None.

    Returns:
        _type_: _description_
    """    

##🚩why the value of classe   is 4 here?
    classes = 4
    if weights is None:
        weights = [1./classes]*classes
    datasets = [create_dataset(data_label,
                               i,
                               fcst_shape=fcst_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat,
                               crop_size=crop_size,
                               rotate=rotate,
                               seed=seed)
                for i in range(classes)]
    
    sampled_ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                      weights=weights, seed=seed).batch(batch_size)

    
    if downsample:
        if return_dic:
            sampled_ds = sampled_ds.map(_dataset_downsampler)
        else:
            sampled_ds = sampled_ds.map(_dataset_downsampler_list)
        
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together

def _dataset_downsampler(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID',
                         name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _dataset_downsampler_list(inputs, constants, outputs):
    image = outputs
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID', name='conv_debug', data_format='NHWC')
    inputs = image
    return inputs, constants, outputs

def _dataset_cropper_dict(inputs, outputs, crop_size, seed=None):
    '''
    Random crop of inputs and outputs
    
    Note: this currently only works when inputs and outputs are all the same dimensions (i.e not downscaling)
    
    '''
    outputs = outputs['output']
    hires_inputs = inputs['hi_res_inputs']
    lores_inputs = inputs['lo_res_inputs']
    
    cropped_lores_input, cropped_hires_input, cropped_output = _dataset_cropper_list(lores_inputs, hires_inputs, outputs, crop_size, seed)
    
    cropped_outputs = {'output': cropped_output}
    cropped_inputs = {'hi_res_inputs': cropped_hires_input,
                      'lo_res_inputs': cropped_lores_input}
    
    return cropped_inputs, cropped_outputs

def _dataset_cropper_list(lores_inputs, hires_inputs, outputs, crop_size, seed=None):
    
    (_, _, lores_channels) = lores_inputs.shape
    (_, _, hires_channels) = hires_inputs.shape
    
    if not seed:
        # Choose random seed (to make sure consistent selection)
        seed = (np.random.randint(1e6), np.random.randint(1e6))
    
    cropped_output = tf.image.stateless_random_crop(outputs, size=[crop_size, crop_size, 1], seed=seed)
    cropped_hires_input = tf.image.stateless_random_crop(hires_inputs, size=[crop_size, crop_size, hires_channels] , seed=seed)
    cropped_lores_input = tf.image.stateless_random_crop(lores_inputs, size=[crop_size, crop_size, lores_channels], seed=seed)
    
    return cropped_lores_input, cropped_hires_input, cropped_output


def _dataset_rotater_list(lores_inputs, hires_inputs, outputs, seed=None):
    
    if seed is not None:
        np.random.seed(seed=seed[0])
    k = np.random.choice([0,1,2,3],1)[0]

    rotated_output = tf.image.rot90(outputs, k=k)
    rotated_hires_input = tf.image.rot90(hires_inputs, k=k)
    rotated_lores_input = tf.image.rot90(lores_inputs, k=k)
    
    return rotated_lores_input, rotated_hires_input, rotated_output

def _dataset_rotater_dict(inputs, outputs, seed=None):
    '''
    Random rotation of inputs and outputs
    '''
    outputs = outputs['output']
    hires_inputs = inputs['hi_res_inputs']
    lores_inputs = inputs['lo_res_inputs']
    
    rotated_lores_input, rotated_hires_input, rotated_output = _dataset_rotater_list(lores_inputs, hires_inputs, outputs, seed=seed)
    
    rotated_outputs = {'output': rotated_output}
    rotated_inputs = {'hi_res_inputs': rotated_hires_input,
                      'lo_res_inputs': rotated_lores_input}
    
    return rotated_inputs, rotated_outputs

def _parse_batch(record_batch,
                 insize=(20, 20, 9),
                 consize=(200, 200, 2),
                 outsize=(200, 200, 1)):
    """_summary_

    Args:
        record_batch (tf.python.framework.ops.EagerTensor): Single item from a tf Records Dataset
        insize (tuple, optional): shape of the forecast data (lat, lon, n_features). Defaults to (20, 20, 9).
        consize (tuple, optional): shape of the constants data (lat, lon, n_features). Defaults to (200, 200, 2).
        outsize (tuple, optional): shape of the output (lat, lon, n_features). Defaults to (200, 200, 1).

    Returns:
        tuple: tuple of dicts, containing inputs and outputs
    """
    # Create a description of the features
    

    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    
    if return_dic:
        output = ({'lo_res_inputs': example['generator_input'],
                   'hi_res_inputs': example['constants']},
                  {'output': example['generator_output']})

        return output
    else:
        return example['generator_input'], example['constants'], example['generator_output']


def create_dataset(data_label: str,
                   clss: str,
                   fcst_shape=(20, 20, 9),
                   con_shape=(200, 200, 2),
                   out_shape=(200, 200, 1),
                   folder: str=records_folder,
                   shuffle_size: int=256,
                   repeat: bool=True,
                   crop_size: int=None,
                   rotate: bool=False,
                   seed: int=None):
    """
    Load tfrecords and parse into appropriate shapes

    Args:
        year (int): year
        clss (str): class (bin to take data from)
        fcst_shape (tuple, optional): shape of the forecast data (lat, lon, n_features). Defaults to (20, 20, 9).
        con_shape (tuple, optional): shape of the constants data (lat, lon, n_features). Defaults to (200, 200, 2).
        out_shape (tuple, optional): shape of the output (lat, lon, n_features). Defaults to (200, 200, 1).
        folder (_type_, optional): folder containing tf records. Defaults to records_folder.
        shuffle_size (int, optional): buffer size for shuffling. Defaults to 1024.
        crop_size (int, optional): Size to crop randomly crop images to.
        rotate (bool, optional): whether to apply random rotations or not
        repeat (bool, optional): create repeat dataset or not. Defaults to True.

    Returns:
        tf.data.DataSet: _description_
    """
    
    if seed:
        if not isinstance(seed, int):
            int_seed = seed[0]
        else:
            int_seed = seed
    else:
        int_seed = None
    
    files_ds = tf.data.Dataset.list_files(f"{folder}/{data_label}_*.{clss}.*.tfrecords")
     
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=AUTOTUNE)
    
    ds = ds.shuffle(shuffle_size, seed=int_seed)
    
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
                # num_parallel_calls=AUTOTUNE)
                
    if crop_size:
        if return_dic:
            ds = ds.map(lambda x,y: _dataset_cropper_dict(x, y, crop_size=crop_size, seed=seed), 
                                                          num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x,y,z: _dataset_cropper_list(x, y, z, crop_size=crop_size, seed=seed), 
                                                            num_parallel_calls=AUTOTUNE)
            
    if rotate:
        if return_dic:
            ds = ds.map(lambda x,y: _dataset_rotater_dict(x, y, seed=seed), 
                                                          num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x,y,z: _dataset_rotater_list(x, y, z, seed=seed), 
                                                            num_parallel_calls=AUTOTUNE)
            
        
    if repeat:
        return ds.repeat()
    else:
        return ds


def create_fixed_dataset(year: int=None,
                         mode: str='validation',
                         batch_size: int=16,
                         downsample: bool=False,
                         fcst_shape: tuple=(20, 20, 9),
                         con_shape: tuple=(200, 200, 2),
                         out_shape: tuple=(200, 200, 1),
                         name: str=None,
                         folder: str=records_folder):
    assert year is not None or name is not None, "Must specify year or file name"
    if name is None:
        name = os.path.join(folder, f"{mode}{year}.tfrecords")
    else:
        name = os.path.join(folder, name)
    fl = glob.glob(name)
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=1)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    ds = ds.batch(batch_size)
    if downsample and return_dic:
        ds = ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        ds = ds.map(_dataset_downsampler_list)
    return ds


def _float_feature(list_of_floats: list):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def write_data(year_month_ranges: list,
               data_label: str,
               hours: list,
               data_config: dict=None,
               debug: bool=False,
               num_shards: int=1) -> str:
    """
    Function to write training data to TF records

    Args:
        year_month_range (list): List of date strings in YYYYMM format, or a list of lists of date strings for non contiguous date ranges. The code will take all dates between the maximum and minimum year months (inclusive)
        data_label (str): Label to assign to data (e.g. train, validate)
        forecast_data_source (str): Source of forecast data (e.g. ifs)
        observational_data_source (str): Source of observational data (e.g. imerg)
        hours (list): List of hours to include
        data_paths (dict, optional): Dict of paths to the data sources. Defaults to DATA_PATHS.
        longitude_range (list, optional): Longitude range to use. Defaults to None.
        debug (bool, optional): Debug mode. Defaults to False.
        config (dict, optional): Config dict. Defaults to None. If None then will read from default config location
        num_shards (int, optional): Number of shards to split each tfrecord into (to make sure records are not too big)

    Returns:
        str: Name of directory that records have been written to
    """

    from .data_generator import DataGenerator
    logger.info('Start of write data')
    logger.info(locals())
    
    if not data_config:
        data_config = read_config.read_config(config_filename='data_config.yaml')
        print('No config provided, reading from default location') ###🚩 add this print statement

    data_paths=get_data_paths(data_config=data_config)
    records_folder = data_paths['TFRecords']["tfrecords_path"]
    if not os.path.isdir(records_folder):
        os.makedirs(records_folder, exist_ok=True)
    
    #  Create directory that is hash of setup params, so that we know it's the right data later on
    data_config_dict = utils.convert_namespace_to_dict(data_config)
    # hash_dir = os.path.join(records_folder, hash_dict(data_config_dict))
    hash_dir = os.path.join(records_folder, 'final_tfrecord/')  ##🚩 change here to final for the time being)
    
    if not os.path.isdir(hash_dir):
        os.makedirs(hash_dir, exist_ok=True)
    
    print(f'Output folder will be {hash_dir}')
        
    # Write params in directory
    write_to_yaml(os.path.join(hash_dir, 'data_config.yaml'), data_config_dict)
    
    with open(os.path.join(hash_dir, 'git_commit.txt'), 'w+') as ofh:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        ofh.write(sha)
    
    if isinstance(year_month_ranges[0], str):
        year_month_ranges = [year_month_ranges]
    
    # Create file handles
    fle_hdles = {}
    for hour in hours:
        fle_hdles[hour] = {}
        for cl in range(data_config.num_classes):
            fle_hdles[hour][cl] = []
            for shard in range(num_shards):
                flename = os.path.join(hash_dir, f"{data_label}_{hour}.{cl}.{shard}.tfrecords")
                fle_hdles[hour][cl].append(tf.io.TFRecordWriter(flename))
    
    for year_month_range in year_month_ranges:  
        
        dates = date_range_from_year_month_range(year_month_range)
        start_date = dates[0]
        dates = [item for item in dates if file_exists(data_source=data_config.fcst_data_source, year=item.year,
                                                            month=item.month, day=item.day,
                                                            data_paths=data_paths)]
        if dates:
            
                
            if not dates[0] == start_date and not debug:
                # Means there likely isn't forecast data for the day before
                dates = dates[1:]
        
            
            if data_config.class_bin_boundaries is not None:
                print(f'Data will be bundled according to class bin boundaries provided: {data_config.class_bin_boundaries}')
                num_class = len(data_config.class_bin_boundaries) + 1

            if debug:
                dates = dates[:1]
                
            print('Hour = ', hour)

            dgc = DataGenerator(data_config=data_config,
                                dates=[item.strftime('%Y%m%d') for item in dates],
                                batch_size=1,
                                shuffle=False,
                                hour=hour)
            print('Fetching batches')
            for batch, date in tqdm(enumerate(dates), total=len(dates), position=0, leave=True):
                
                logger.debug(f"hour={hour}, batch={batch}")
                
                try:
                    
                    sample = dgc.__getitem__(batch)
                    (depth, width, height) = sample[1]['output'].shape
                
                    for k in range(depth):
                    
                        observations = sample[1]['output'][k,...].flatten()
                        forecast = sample[0]['lo_res_inputs'][k,...].flatten()
                        const = sample[0]['hi_res_inputs'][k,...].flatten()

                        # Check no Null values
                        if np.isnan(observations).any() or np.isnan(forecast).any() or np.isnan(const).any():
                            raise ValueError('Unexpected NaN values in data')
                        
                        # Check for empty data
                        if forecast.sum() == 0 or const.sum() == 0:
                            raise ValueError('one or more of arrays is all zeros')
                        
                        # Check hi res data has same dimensions
                            
                        feature = {
                            'generator_input': _float_feature(forecast),
                            'constants': _float_feature(const),
                            'generator_output': _float_feature(observations)
                        }

                        features = tf.train.Features(feature=feature)
                        example = tf.train.Example(features=features)
                        example_to_string = example.SerializeToString()
                        
                        # If provided, bin data according to bin boundaries (typically quartiles)
                        if data_config.class_bin_boundaries is not None:
                                                    
                            threshold = 0.1
                            if data_config.output_normalisation is not None:
                                rainy_pixel_fraction = (denormalise(observations, normalisation_type=data_config.output_normalisation) > threshold).mean()
                            else:
                                rainy_pixel_fraction = (observations > threshold).mean()

                            clss = np.digitize(rainy_pixel_fraction, right=False)
                            
                        else:
                            clss = random.choice(range(data_config.num_classes))
                            
                        # Choose random shard
                        fh = random.choice(fle_hdles[hour][clss])

                        fh.write(example_to_string)
                
                except FileNotFoundError as e:
                    print(f"Error loading hour={hour}, date={date}")
                    
                
        else:
            print('No dates found')
            
    for hour in hours:
        for cl, fhs in fle_hdles[hour].items():
            for fh in fhs:
                fh.close()
    
                            
    return hash_dir




# import os
# import logging
# from google.cloud import storage
# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
# import git
# from dsrnngan.utils import read_config, utils
# from dsrnngan.data.data import file_exists, denormalise
# from dsrnngan.utils.utils import hash_dict, write_to_yaml, date_range_from_year_month_range
# from dsrnngan.utils.read_config import get_data_paths

# logger = logging.getLogger(__file__)
# logger.setLevel(logging.DEBUG)

# def write_data(year_month_ranges: list,
#                data_label: str,
#                hours: list,
#                data_config: dict = None,
#                debug: bool = False,
#                num_shards: int = 1) -> str:
#     """
#     Function to write training data to TF records in GCS.

#     Args:
#         year_month_range (list): List of date strings in YYYYMM format, or a list of lists of date strings for non contiguous date ranges. The code will take all dates between the maximum and minimum year months (inclusive)
#         data_label (str): Label to assign to data (e.g. train, validate)
#         hours (list): List of hours to include
#         data_config (dict, optional): Config dict. Defaults to None. If None then will read from default config location
#         debug (bool, optional): Debug mode. Defaults to False.
#         num_shards (int, optional): Number of shards to split each tfrecord into (to make sure records are not too big)

#     Returns:
#         str: GCS path of the directory that records have been written to
#     """

#     from dsrnngan.data.data_generator import DataGenerator

#     logger.info('Start of write data')
#     logger.info(locals())

#     if not data_config:
#         data_config = read_config.read_config(config_filename='data_config.yaml')

#     data_paths = get_data_paths(data_config=data_config)
#     records_folder = data_paths['TFRecords']["tfrecords_path"]  # e.g., gs://your-bucket/ngcm/tfrecords
#     bucket_name = records_folder.replace("gs://", "").split("/")[0]
#     prefix = "/".join(records_folder.replace("gs://", "").split("/")[1:])  # e.g., ngcm/tfrecords

#     # Initialize GCS client
#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)

#     # Create directory that is hash of setup params in GCS
#     data_config_dict = utils.convert_namespace_to_dict(data_config)
#     hash_dir = f"{prefix}/{hash_dict(data_config_dict)}" if prefix else hash_dict(data_config_dict)
#     if not blob_exists(bucket, hash_dir):
#         bucket.blob(f"{hash_dir}/").upload_from_string("")  # Create "directory" (empty blob)

#     logger.info(f'Output folder will be gs://{bucket_name}/{hash_dir}')

#     # Write params in GCS
#     write_to_yaml_gcs(bucket, f"{hash_dir}/data_config.yaml", data_config_dict)

#     # Write git commit hash to GCS
#     with open_gcs_file(bucket, f"{hash_dir}/git_commit.txt", 'w+') as ofh:
#         repo = git.Repo(search_parent_directories=True)
#         sha = repo.head.object.hexsha
#         ofh.write(sha)

#     if isinstance(year_month_ranges[0], str):
#         year_month_ranges = [year_month_ranges]

#     # Create file handles for GCS
#     fle_hdles = {}
#     for hour in hours:
#         fle_hdles[hour] = {}
#         for cl in range(data_config.num_classes):
#             fle_hdles[hour][cl] = []
#             for shard in range(num_shards):
#                 gcs_filename = f"gs://{bucket_name}/{hash_dir}/{data_label}_{hour}.{cl}.{shard}.tfrecords"
#                 fle_hdles[hour][cl].append(tf.io.TFRecordWriter(gcs_filename))

#     for year_month_range in year_month_ranges:
#         dates = date_range_from_year_month_range(year_month_range)
#         start_date = dates[0]
#         dates = [item for item in dates if file_exists(data_source=data_config.fcst_data_source, year=item.year,
#                                                        month=item.month, day=item.day,
#                                                        data_paths=data_paths)]
#         if dates:
#             if not dates[0] == start_date and not debug:
#                 dates = dates[1:]

#             if data_config.class_bin_boundaries is not None:
#                 logger.info(f'Data will be bundled according to class bin boundaries provided: {data_config.class_bin_boundaries}')
#                 num_class = len(data_config.class_bin_boundaries) + 1

#             if debug:
#                 dates = dates[:1]

#             logger.info('Hour = %s', hour)

#             dgc = DataGenerator(data_config=data_config,
#                                 dates=[item.strftime('%Y%m%d') for item in dates],
#                                 batch_size=1,
#                                 shuffle=False,
#                                 hour=hour)
#             logger.info('Fetching batches')
#             for batch, date in tqdm(enumerate(dates), total=len(dates), position=0, leave=True):
#                 logger.debug(f"hour={hour}, batch={batch}")

#                 try:
#                     sample = dgc.__getitem__(batch)
#                     (depth, width, height) = sample[1]['output'].shape

#                     for k in range(depth):
#                         observations = sample[1]['output'][k, ...].flatten()
#                         forecast = sample[0]['lo_res_inputs'][k, ...].flatten()
#                         const = sample[0]['hi_res_inputs'][k, ...].flatten()

#                         # Check no Null values
#                         if np.isnan(observations).any() or np.isnan(forecast).any() or np.isnan(const).any():
#                             raise ValueError('Unexpected NaN values in data')

#                         # Check for empty data
#                         if forecast.sum() == 0 or const.sum() == 0:
#                             raise ValueError('one or more of arrays is all zeros')

#                         feature = {
#                             'generator_input': _float_feature(forecast),
#                             'constants': _float_feature(const),
#                             'generator_output': _float_feature(observations)
#                         }

#                         features = tf.train.Features(feature=feature)
#                         example = tf.train.Example(features=features)
#                         example_to_string = example.SerializeToString()

#                         # If provided, bin data according to bin boundaries
#                         if data_config.class_bin_boundaries is not None:
#                             threshold = 0.1
#                             if data_config.output_normalisation is not None:
#                                 rainy_pixel_fraction = (denormalise(observations, normalisation_type=data_config.output_normalisation) > threshold).mean()
#                             else:
#                                 rainy_pixel_fraction = (observations > threshold).mean()
#                             clss = np.digitize(rainy_pixel_fraction, bins=data_config.class_bin_boundaries, right=False)
#                         else:
#                             clss = random.choice(range(data_config.num_classes))

#                         # Choose random shard
#                         fh = random.choice(fle_hdles[hour][clss])
#                         fh.write(example_to_string)

#                 except FileNotFoundError as e:
#                     logger.error(f"Error loading hour={hour}, date={date}")

#         else:
#             logger.info('No dates found')

#     for hour in hours:
#         for cl, fhs in fle_hdles[hour].items():
#             for fh in fhs:
#                 fh.close()

#     return f"gs://{bucket_name}/{hash_dir}"

# Helper functions for GCS
def write_to_yaml_gcs(bucket, blob_name, data):
    blob = bucket.blob(blob_name)
    yaml_content = utils.convert_dict_to_yaml(data)
    blob.upload_from_string(yaml_content)

def open_gcs_file(bucket, blob_name, mode='r'):
    from io import BytesIO
    blob = bucket.blob(blob_name)
    if mode == 'w+':
        return BytesIO()  # Will be written to GCS when closed
    else:
        return BytesIO(blob.download_as_bytes())

def blob_exists(bucket, blob_name):
    return bucket.get_blob(blob_name) is not None

def write_train_test_data(*args, training_range,
                          validation_range=None,
                          test_range=None, **kwargs):
    
    
    write_data(training_range, *args,
               data_label='train', **kwargs)
    
    if validation_range:
        write_data(training_range, *args,
               data_label='train', **kwargs)
        
    if test_range:
        print('\n*** Writing test data')

        write_data(test_range, *args,
                   data_label='test', **kwargs)


def save_dataset(tfrecords_dataset, flename, max_batches=None):

    assert return_dic, "Only works with return_dic=True"
    flename = os.path.join(records_folder, flename)
    fle_hdle = tf.io.TFRecordWriter(flename)
    for ii, sample in enumerate(tfrecords_dataset):
        logger.info(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k, ...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Write data to tf records.')

    parser.add_argument('--fcst-hours', nargs='+', default=np.arange(24), type=int, 
                    help='Hour(s) to process (space separated)')
    parser.add_argument('--records-folder', type=str, default=None)
    parser.add_argument('--data-config-path', type=str, default=None)
    parser.add_argument('--model-config-path', type=str, default=None)
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()
    
    # Load relevant parameters from local config
    if args.data_config_path:
        path_split = os.path.split(args.data_config_path)
        data_config = read_config.read_data_config(config_filename=path_split[-1],
                                                     config_folder=path_split[0])
    else:
        data_config = read_config.read_data_config()
    
    if args.data_config_path:
        path_split = os.path.split(args.model_config_path)
        model_config = read_config.read_model_config(config_filename=path_split[-1],
                                                     config_folder=path_split[0])
    else:
        model_config = read_config.read_model_config()
    
    val_range = None
    eval_range = None
     
    if args.debug:
        training_range = ['201701']
    else:
        training_range = model_config.train.training_range
        
        if hasattr(model_config.val, 'val_range'):
            val_range = model_config.val.val_range
        
        if hasattr(model_config, 'eval'):
            if hasattr(model_config.eval, 'eval_range'):
                eval_range =  model_config.eval.eval_range
    
    write_train_test_data(training_range=training_range,
                            validation_range=val_range,
                            test_range=eval_range,
                            data_config=data_config,
                            hours=args.fcst_hours)
