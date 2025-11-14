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

# TODO: pass this as function argument instead
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

# Use autotune to tune the prefetching of records in parallel to processing to improve performance
AUTOTUNE = tf.data.AUTOTUNE


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
                         repeat: bool=False,
                         downsample: bool=False,
                         folder: str=records_folder,
                         shuffle_size: int=256,
                         weights: list=None,
                         crop_size: int=None,
                         rotate: bool=False,
                         seed: int=None):
    """
    Create a mixed dataset from multiple classes with weighted sampling.

    Args:
        data_label (str): Label of data
        batch_size (int): size of batches
        fcst_shape (tuple[int, int, int]): shape of forecast input
        con_shape (tuple[int, int, int]): shape of constants input
        out_shape (tuple[int, int, int]): shape of output
        repeat (bool, optional): repeat dataset or not. Defaults to False.
        downsample (bool, optional): whether to downsample or not. Defaults to False.
        folder (str, optional): folder containing tf records. Defaults to records_folder.
        shuffle_size (int, optional): buffer size of shuffling. Defaults to 256.
        weights (list, optional): list of floats, weights of classes when sampling. Defaults to None.
        crop_size (int, optional): Size to randomly crop images to.
        rotate (bool, optional): If true then data is augmented by random rotation.
        seed (int, optional): seed for shuffling and sampling. Defaults to None.

    Returns:
        tf.data.Dataset: Mixed dataset
    """
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
                                                      weights=weights, 
                                                      seed=seed).batch(batch_size)
    
    if downsample:
        if return_dic:
            sampled_ds = sampled_ds.map(_dataset_downsampler)
        else:
            sampled_ds = sampled_ds.map(_dataset_downsampler_list)
    
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds


def _dataset_downsampler(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], 
                         padding='VALID', name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _dataset_downsampler_list(inputs, constants, outputs):
    image = outputs
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], 
                         padding='VALID', name='conv_debug', data_format='NHWC')
    inputs = image
    return inputs, constants, outputs


def _dataset_cropper_dict(inputs, outputs, crop_size, seed=None):
    """
    Random crop of inputs and outputs
    
    Note: this currently only works when inputs and outputs are all the same dimensions 
    (i.e not downscaling)
    """
    outputs = outputs['output']
    hires_inputs = inputs['hi_res_inputs']
    lores_inputs = inputs['lo_res_inputs']
    
    cropped_lores_input, cropped_hires_input, cropped_output = _dataset_cropper_list(
        lores_inputs, hires_inputs, outputs, crop_size, seed)
    
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
    
    cropped_output = tf.image.stateless_random_crop(outputs, 
                                                     size=[crop_size, crop_size, 1], 
                                                     seed=seed)
    cropped_hires_input = tf.image.stateless_random_crop(hires_inputs, 
                                                          size=[crop_size, crop_size, hires_channels], 
                                                          seed=seed)
    cropped_lores_input = tf.image.stateless_random_crop(lores_inputs, 
                                                          size=[crop_size, crop_size, lores_channels], 
                                                          seed=seed)
    
    return cropped_lores_input, cropped_hires_input, cropped_output


def _dataset_rotater_list(lores_inputs, hires_inputs, outputs, seed=None):
    if seed is not None:
        np.random.seed(seed=seed[0])
    k = np.random.choice([0, 1, 2, 3], 1)[0]

    rotated_output = tf.image.rot90(outputs, k=k)
    rotated_hires_input = tf.image.rot90(hires_inputs, k=k)
    rotated_lores_input = tf.image.rot90(lores_inputs, k=k)
    
    return rotated_lores_input, rotated_hires_input, rotated_output


def _dataset_rotater_dict(inputs, outputs, seed=None):
    """Random rotation of inputs and outputs"""
    outputs = outputs['output']
    hires_inputs = inputs['hi_res_inputs']
    lores_inputs = inputs['lo_res_inputs']
    
    rotated_lores_input, rotated_hires_input, rotated_output = _dataset_rotater_list(
        lores_inputs, hires_inputs, outputs, seed=seed)
    
    rotated_outputs = {'output': rotated_output}
    rotated_inputs = {'hi_res_inputs': rotated_hires_input,
                      'lo_res_inputs': rotated_lores_input}
    
    return rotated_inputs, rotated_outputs


def _parse_batch(record_batch,
                 insize=(20, 20, 9),
                 consize=(200, 200, 2),
                 outsize=(200, 200, 1)):
    """
    Parse a batch of TFRecords

    Args:
        record_batch: Single item from a tf Records Dataset
        insize (tuple, optional): shape of the forecast data (lat, lon, n_features)
        consize (tuple, optional): shape of the constants data (lat, lon, n_features)
        outsize (tuple, optional): shape of the output (lat, lon, n_features)

    Returns:
        tuple: tuple of dicts, containing inputs and outputs
    """
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


def create_dataset(
    data_label: str,
    clss: str,
    years: list = None,
    fcst_shape=(20, 20, 9),
    con_shape=(200, 200, 2),
    out_shape=(200, 200, 1),
    folder: str=records_folder,
    shuffle_size: int = 256,
    repeat: bool = False,
    crop_size: int = None,
    rotate: bool = False,
    seed: int = None,
):
    """
    Load TFRecords across multiple folders and filter by years.
    
    Args:
        data_label (str): Label of data
        clss (str): class (bin to take data from)
        years (list): list of int, e.g. [2000, 2001] for training
        fcst_shape (tuple): shape of the forecast data (lat, lon, n_features)
        con_shape (tuple): shape of the constants data (lat, lon, n_features)
        out_shape (tuple): shape of the output (lat, lon, n_features)
        folder (str): folder containing tf records
        shuffle_size (int): buffer size for shuffling
        repeat (bool): create repeat dataset or not
        crop_size (int): Size to randomly crop images to
        rotate (bool): whether to apply random rotations or not
        seed (int): seed for shuffling and sampling
    
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Convert crop_size
    if isinstance(crop_size, str):
        if crop_size.lower() == "none":
            crop_size = None
        else:
            try:
                crop_size = int(crop_size)
            except ValueError:
                print(f"Warning: invalid crop_size '{crop_size}', setting to None")
                crop_size = None

    # Convert seed
    if isinstance(seed, str):
        if seed.lower() == "none":
            seed = None
        else:
            try:
                seed = int(seed)
            except ValueError:
                print(f"Warning: invalid seed '{seed}', setting to None")
                seed = None

    int_seed = seed if isinstance(seed, int) else (seed[0] if seed else None)

    # Folder can be string or list
    folder_list = [folder] if isinstance(folder, str) else list(folder)

    # Gather all TFRecord files matching the years
    all_files = []
    for fdir in folder_list:
        if years is None:
            # All train_*.tfrecords files in all subdirectories
            pattern = os.path.join(fdir, "**", f"{data_label}_*.tfrecords")
            all_files.extend(glob.glob(pattern, recursive=True))
        else:
            for yr in years:
                # Search in the year subdirectory only
                pattern = os.path.join(fdir, str(yr), f"{data_label}_*.tfrecords")
                all_files.extend(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No TFRecords found in {folder_list} for years {years}")

    files_ds = tf.data.Dataset.from_tensor_slices(all_files)
    files_ds = files_ds.shuffle(len(all_files), seed=int_seed)

    ds = tf.data.TFRecordDataset(files_ds, num_parallel_reads=AUTOTUNE)

    # Parse
    ds = ds.map(
        lambda x: _parse_batch(x, insize=fcst_shape, consize=con_shape, outsize=out_shape),
        num_parallel_calls=AUTOTUNE,
    )

    # Crop
    if crop_size:
        if return_dic:
            ds = ds.map(
                lambda x, y: _dataset_cropper_dict(x, y, crop_size=crop_size, seed=seed),
                num_parallel_calls=AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda x, y, z: _dataset_cropper_list(x, y, z, crop_size=crop_size, seed=seed),
                num_parallel_calls=AUTOTUNE,
            )

    # Rotate
    if rotate:
        if return_dic:
            ds = ds.map(
                lambda x, y: _dataset_rotater_dict(x, y, seed=seed),
                num_parallel_calls=AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda x, y, z: _dataset_rotater_list(x, y, z, seed=seed),
                num_parallel_calls=AUTOTUNE,
            )

    return ds.repeat() if repeat else ds


def create_fixed_dataset(
    years: list = None,
    mode: str = 'validation',
    batch_size: int = 2,
    downsample: bool = False,
    fcst_shape: tuple = (20, 20, 9),
    con_shape: tuple = (200, 200, 2),
    out_shape: tuple = (200, 200, 1),
    folder: str = None,
    data_label: str = 'train',
    clss: str = 'storm',
):
    """
    Create a fixed dataset (batch) for validation or testing.
    Selection is done via `years`.
    """
    folder_list = [folder] if isinstance(folder, str) else list(folder)

    all_files = []
    for fdir in folder_list:
        if years is None:
            pattern = f"{data_label}_*.{clss}.*.tfrecords"
            all_files.extend(glob.glob(os.path.join(fdir, pattern)))
        else:
            for yr in years:
                pattern = f"{data_label}_{yr}_*.{clss}.*.tfrecords"
                all_files.extend(glob.glob(os.path.join(fdir, pattern)))

    if not all_files:
        raise FileNotFoundError(f"No TFRecords found in {folder_list} for years {years}")

    files_ds = tf.data.Dataset.from_tensor_slices(all_files)
    ds = tf.data.TFRecordDataset(files_ds, num_parallel_reads=1)

    ds = ds.map(
        lambda x: _parse_batch(x, insize=fcst_shape, consize=con_shape, outsize=out_shape),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(batch_size)

    if downsample:
        if return_dic:
            ds = ds.map(_dataset_downsampler)
        else:
            ds = ds.map(_dataset_downsampler_list)

    return ds


def _float_feature(list_of_floats: list):
    """Create float feature for TFRecords"""
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
        year_month_ranges (list): List of date strings in YYYYMM format, or a list of lists
        data_label (str): Label to assign to data (e.g. train, validate)
        hours (list): List of hours to include
        data_config (dict, optional): Config dict. If None then will read from default config
        debug (bool, optional): Debug mode. Defaults to False.
        num_shards (int, optional): Number of shards to split each tfrecord into

    Returns:
        str: Name of directory that records have been written to
    """
    from .data_generator import DataGenerator
    logger.info('Start of write data')
    logger.info(locals())
    
    if not data_config:
        data_config = read_config.read_config(config_filename='data_config.yaml')
        print('No config provided, reading from default location')

    data_paths = get_data_paths(data_config=data_config)
    records_folder = data_paths['TFRecords']["tfrecords_path"]
    if not os.path.isdir(records_folder):
        os.makedirs(records_folder, exist_ok=True)
    
    # Create directory that is hash of setup params
    data_config_dict = utils.convert_namespace_to_dict(data_config)
    hash_dir = os.path.join(records_folder, 'final_tfrecord/')
    
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
        dates = [item for item in dates if file_exists(data_source=data_config.fcst_data_source, 
                                                        year=item.year,
                                                        month=item.month, 
                                                        day=item.day,
                                                        data_paths=data_paths)]
        if dates:
            if not dates[0] == start_date and not debug:
                dates = dates[1:]
            
            if data_config.class_bin_boundaries is not None:
                print(f'Data will be bundled according to class bin boundaries provided: '
                      f'{data_config.class_bin_boundaries}')
                num_class = len(data_config.class_bin_boundaries) + 1

            if debug:
                dates = dates[:1]
            
            # Move the hour loop here to process each hour
            for hour in hours:
                print(f'Hour = {hour}')
                
                dgc = DataGenerator(data_config=data_config,
                                    dates=[item.strftime('%Y%m%d') for item in dates],
                                    batch_size=1,
                                    shuffle=False,
                                    hour=hour)
                print('Fetching batches')
                dates_subset = np.random.choice(np.asarray(dates), size=100, replace=False)
                for batch, date in tqdm(enumerate(dates_subset), 
                                        total=len(dates_subset), 
                                        position=0, 
                                        leave=True):
                    logger.debug(f"hour={hour}, batch={batch}")
                    
                    try:
                        sample = dgc.__getitem__(batch)
                        (depth, width, height) = sample[1]['output'].shape
                        
                        for k in range(depth):
                            observations = sample[1]['output'][k, ...].flatten()
                            forecast = sample[0]['lo_res_inputs'][k, ...].flatten()
                            const = sample[0]['hi_res_inputs'][k, ...].flatten()

                            # Check no Null values
                            if (np.isnan(observations).any() or 
                                np.isnan(forecast).any() or 
                                np.isnan(const).any()):
                                raise ValueError('Unexpected NaN values in data')
                            
                            # Check for empty data
                            if forecast.sum() == 0 or const.sum() == 0:
                                raise ValueError('one or more of arrays is all zeros')
                            
                            feature = {
                                'generator_input': _float_feature(forecast),
                                'constants': _float_feature(const),
                                'generator_output': _float_feature(observations)
                            }

                            features = tf.train.Features(feature=feature)
                            example = tf.train.Example(features=features)
                            example_to_string = example.SerializeToString()
                            
                            # If provided, bin data according to bin boundaries
                            if data_config.class_bin_boundaries is not None:
                                threshold = 0.1
                                if data_config.output_normalisation is not None:
                                    rainy_pixel_fraction = (
                                        denormalise(observations, 
                                                    normalisation_type=data_config.output_normalisation) 
                                        > threshold).mean()
                                else:
                                    rainy_pixel_fraction = (observations > threshold).mean()

                                clss = np.digitize(rainy_pixel_fraction, 
                                                   bins=data_config.class_bin_boundaries, 
                                                   right=False)
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


def write_to_yaml_gcs(bucket, blob_name, data):
    """Write YAML data to GCS"""
    blob = bucket.blob(blob_name)
    yaml_content = utils.convert_dict_to_yaml(data)
    blob.upload_from_string(yaml_content)


def open_gcs_file(bucket, blob_name, mode='r'):
    """Open a file in GCS"""
    from io import BytesIO
    blob = bucket.blob(blob_name)
    if mode == 'w+':
        return BytesIO()  # Will be written to GCS when closed
    else:
        return BytesIO(blob.download_as_bytes())


def blob_exists(bucket, blob_name):
    """Check if a blob exists in GCS"""
    return bucket.get_blob(blob_name) is not None


def write_train_test_data(*args, 
                          training_range,
                          validation_range=None,
                          test_range=None, 
                          **kwargs):
    """Write training, validation, and test data"""
    write_data(training_range, *args, data_label='train', **kwargs)
    
    # if validation_range:
    #     print('\n*** Writing validation data')
    #     write_data(validation_range, *args, data_label='validation', **kwargs)
    
    # if test_range:
    #     print('\n*** Writing test data')
    #     write_data(test_range, *args, data_label='test', **kwargs)


def save_dataset(tfrecords_dataset, flename, max_batches=None):
    """Save a TFRecords dataset to file"""
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
                'generator_input': _float_feature(
                    sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(
                    sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(
                    sample[1]['output'][k, ...].numpy().flatten())
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
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    # Load relevant parameters from local config
    if args.data_config_path:
        path_split = os.path.split(args.data_config_path)
        data_config = read_config.read_data_config(config_filename=path_split[-1],
                                                    config_folder=path_split[0])
    else:
        data_config = read_config.read_data_config()
    
    if args.model_config_path:
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
                eval_range = model_config.eval.eval_range
    
    write_train_test_data(training_range=training_range,
                          validation_range=val_range,
                          test_range=eval_range,
                          data_config=data_config,
                          hours=args.fcst_hours)