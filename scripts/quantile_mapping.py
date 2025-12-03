"""
Script to quantile map the forecasts

Note that this requires data to be prepared from the training set, and so is currently
not in a fit state to be used without a proper pipeline being set up 

lbatch -t 10 -m 100 --queue short --conda-env base -a ID --array-range 1 10 --cmd python -m scripts.quantile_mapping --num-lat-lon-chunks ARRAY_ID --model-type cropped_4000 --output-folder /user/work/uz22147/quantile_mapping


"""

import pickle
import os, sys
import copy
import numpy as np
from pathlib import Path
from glob import glob
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
sys.path.insert(1, str(HOME / 'dsrnngan'))

from dsrnngan.evaluation.benchmarks import QuantileMapper, quantile_map_grid
from dsrnngan.utils import read_config
from dsrnngan.evaluation.plots import plot_quantiles ,get_quantile_data ,range_dict

parser = ArgumentParser(description='Script for quantile mapping.')
parser.add_argument('--model-eval-folder', type=str, help='Folder containing evaluated samples for the model', required=True)
parser.add_argument('--model-number', type=str, help='Checkpoint number of model that created the samples', required=True)
parser.add_argument('--num-lat-lon-chunks', type=int, help='Number of chunks to split up spatial data into along each axis', default=1)
parser.add_argument('--output-folder', type=str, help='Folder to save plots in')
parser.add_argument('--min-points-per-quantile', type=int, default=1, help='Minimum number of data points per quantile in the plots')
parser.add_argument('--save-data', action='store_true', help='Save the quantile mapped data to the model folder')
parser.add_argument('--save-qmapper', action='store_true', help='Save the quantile mapping objects to the model folder')
parser.add_argument('--plot', action='store_true', help='Make plots')
parser.add_argument('--debug', action='store_true', help='Debug mode')
args = parser.parse_args()

if not args.plot and not args.save_data and not args.save_qmapper:
    raise ValueError('Either --plot, --save-data or --save-qmapper must be specified')

###########################
# Load model data
###########################

print('Loading model data', flush=True)

# Get best model
model_eval_folder = args.model_eval_folder
model_number =args.model_number
base_folder = '/'.join(model_eval_folder.split('/')[:-1]) 

with open(os.path.join(model_eval_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

if args.debug:
    n_samples = 100
else:
    n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth'][:n_samples, :, :]
samples_gen_array = arrays['samples_gen'][:n_samples, ...]

if len(samples_gen_array.shape) == 3:
    samples_gen_array = np.expand_dims(samples_gen_array, axis=-1)
elif len(samples_gen_array.shape) == 4:
    samples_gen_array = samples_gen_array[...,:1]
    
fcst_array = arrays['fcst_array'][:n_samples, :,: ]
ensmean_array = np.mean(samples_gen_array, axis=-1)[:n_samples, :,:]
dates = [d[0] for d in arrays['dates']][:n_samples]
hours = [h[0] for h in arrays['hours']][:n_samples]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape

print(f'Loaded {n_samples} samples of size {width}x{height} with ensemble size {ensemble_size}', flush=True)
###########################

###########################

# Get lat/lon range from log folder
data_config = read_config.read_data_config(config_folder=base_folder)
model_config = read_config.read_model_config(config_folder=base_folder)

# Locations
latitude_range, longitude_range=read_config.get_lat_lon_range_from_config(data_config=data_config)

lat_range_list = [np.round(item, 2) for item in sorted(latitude_range)]
lon_range_list = [np.round(item, 2) for item in sorted(longitude_range)]

special_areas = {'Lake Victoria': {'lat_range': [-3.05,0.95], 'lon_range': [31.55, 34.55], 'color': 'red'},
                 'Coastal Kenya/Somalia': {'lat_range': [-4.65, 5.45], 'lon_range': [38.85, 48.3], 'color': 'black'},
                 'West EA Rift': {'lat_range': [-4.70,0.30], 'lon_range': [28.25,31.3], 'color': 'green'},
                 'East EA Rift': {'lat_range': [-3.15, 1.55], 'lon_range': [33.85,36.55], 'color': 'purple'},
                 'NW Ethiopian Highlands': {'lat_range': [6.10, 14.15], 'lon_range': [34.60, 40.30], 'color': 'blue'}}

for k, v in special_areas.items():
    lat_vals = [lt for lt in lat_range_list if v['lat_range'][0] <= lt <= v['lat_range'][1]]
    lon_vals = [ln for ln in lon_range_list if v['lon_range'][0] <= ln <= v['lon_range'][1]]
    
    if lat_vals and lon_vals:
 
        special_areas[k]['lat_index_range'] = [lat_range_list.index(lat_vals[0]), lat_range_list.index(lat_vals[-1])]
        special_areas[k]['lon_index_range'] = [lon_range_list.index(lon_vals[0]), lon_range_list.index(lon_vals[-1])]
        
quantile_data_dicts = {'test': {
                    'GAN': samples_gen_array[:, :, :, 0],
                    'Fcst': fcst_array,
                    'Obs (IMERG)': truth_array,
                    'Fcst + qmap': None,
                    'GAN + qmap': None,
                    }}

###########################
# Load training data
###########################

# NOTE:This requires data collection for the model 

# 1. Use glob to find all paths within base_folder that match the 'n*' pattern.
all_n_paths = glob.glob(os.path.join(base_folder, 'n*'))

# 2. Assign fps a list containing the first found path, or an empty list if none were found.
fps = [all_n_paths[0]] if all_n_paths else []

imerg_training_data = []
cgan_training_data = []
ifs_training_data = []
training_dates = []
training_hours = []

for fp in fps:
    with open(os.path.join(fp, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
        arrays = pickle.load(ifh)
    
    if args.debug:
        n_samples = 100
    else:
        n_samples = arrays['truth'].shape[0]
        
    imerg_training_data.append(arrays['truth'][:n_samples,:,:])
    cgan_training_data.append(arrays['samples_gen'][:n_samples,:,:,0])
    ifs_training_data.append(arrays['fcst_array'][:n_samples,:,:])
    training_dates += [d[0] for d in arrays['dates']][:n_samples]
    training_hours += [h[0] for h in arrays['hours']][:n_samples]
    
imerg_training_data = np.concatenate(imerg_training_data, axis=0)
cgan_training_data = np.concatenate(cgan_training_data, axis=0)
ifs_training_data = np.concatenate(ifs_training_data, axis=0)


###########################
print('## Quantile mapping for IFS', flush=True)
###########################
# # Use data prepared by the dsrnngan.data.setupdata script
# fps = glob('/user/work/uz22147/quantile_training_data/*_744.pkl')

# imerg_train_data = []
# ifs_train_data = []
# training_dates = []
# training_hours = []

# if args.debug:
#     fps = fps[:3]

# for fp in tqdm(fps, file=sys.stdout):
#     with open(fp, 'rb') as ifh:
#         training_data = pickle.load(ifh)
    
#     if data_config.output_normalisation is not None:
#         training_data['obs'] = denormalise(training_data['obs'], normalisation_type=data_config.output_normalisation)
#     imerg_train_data.append(training_data['obs'])
    
#     if data_config.normalise_inputs:
#         training_data['fcst_array'] = denormalise(training_data['fcst_array'], normalisation_type=data_config.input_normalisation_strategy['tp']['normalisation'])
#     ifs_train_data.append(training_data['fcst_array'])

#     training_dates += [item[0] for item in training_data['dates']]
#     training_hours += [item[0] for item in training_data['hours']]

# # Need to account for difference in lat/lon ranges; setupdata incorporates the final lat value whereas 
# # the generated data doesn't
# imerg_train_data = np.concatenate(imerg_train_data, axis=0)[:,:-1,:-1]
# ifs_train_data = np.concatenate(ifs_train_data, axis=0)[:,:-1,:-1]



month_ranges = [[n for n in range(1,13)]]


fcst_qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range,
                         num_lat_lon_chunks=args.num_lat_lon_chunks)

# Get auto spaced quantiles, up to one data point per quantile
ifs_quantile_locs = fcst_qmapper.update_quantile_locations(input_data=ifs_training_data, max_step_size=0.01)


if args.num_lat_lon_chunks == 0:
    # Do quantile mapping grid cell by grid cell
    print('Quantile mapping Fcst at grid level')
    quantile_data_dicts['test']['Fcst + qmap']= quantile_map_grid(array_to_correct=fcst_array, 
                                                                                fcst_train_data=ifs_training_data, 
                                                                                obs_train_data=imerg_training_data, 
                                                                                quantiles=ifs_quantile_locs,
                                                                                extrapolate='constant')
    fcst_corrected_train = [] # Not yet implemented for this, as not currently required
else:
    
    fcst_qmapper.train(fcst_data=ifs_training_data, obs_data=imerg_training_data, training_dates=training_dates, training_hours=training_hours)
    quantile_data_dicts['test']['Fcst + qmap'] = fcst_qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)

    fcst_corrected_train = fcst_qmapper.get_quantile_mapped_forecast(fcst=ifs_training_data, dates=training_dates, hours=training_hours)

###########################
print('## Quantile mapping for GAN', flush=True)
###########################


cgan_qmapper = QuantileMapper(month_ranges=month_ranges, latitude_range=latitude_range, longitude_range=longitude_range,
                         num_lat_lon_chunks=args.num_lat_lon_chunks)

# Get auto spaced quantiles, up to one data point per quantile
cgan_quantile_locs = cgan_qmapper.update_quantile_locations(input_data=cgan_training_data, max_step_size=0.01)


if args.num_lat_lon_chunks == 0:
    if args.save_data:
        cgan_corrected = np.empty(samples_gen_array.shape)
        
        for en in range(ensemble_size):
            cgan_corrected[:,:,:,en] =  quantile_map_grid(array_to_correct=samples_gen_array[:,:,:,en], 
                                                                                fcst_train_data=cgan_training_data, 
                                                                            obs_train_data=imerg_training_data, 
                                                                            quantiles=cgan_quantile_locs,
                                                                            extrapolate='constant')
        quantile_data_dicts['test']['GAN + qmap'] = cgan_corrected[:,:,:,0]
    else:
        # Do quantile mapping grid cell by grid cell
        quantile_data_dicts['test']['GAN + qmap'] = quantile_map_grid(array_to_correct=samples_gen_array[:,:,:,0], 
                                                                                fcst_train_data=cgan_training_data, 
                                                                                obs_train_data=imerg_training_data, 
                                                                                quantiles=cgan_quantile_locs,
                                                                            extrapolate='constant')

else:
     
    cgan_qmapper.train(fcst_data=cgan_training_data, obs_data=imerg_training_data, training_dates=training_dates, training_hours=training_hours)
    if args.save_data:
        # Only correct all ensemble members if we are saving data; not needed if just plotting
        cgan_corrected = np.empty(samples_gen_array.shape)
        
        for en in range(ensemble_size):
            cgan_corrected[:,:,:,en] = cgan_qmapper.get_quantile_mapped_forecast(fcst=samples_gen_array[:,:,:,en], dates=dates, hours=hours)
            quantile_data_dicts['test']['GAN + qmap'] = cgan_corrected[...,0]
    else:
        cgan_corrected = quantile_data_dicts['test']['GAN + qmap'] = cgan_qmapper.get_quantile_mapped_forecast(fcst=samples_gen_array[:,:,:,0], dates=dates, hours=hours)
        quantile_data_dicts['test']['GAN + qmap'] = cgan_corrected



if args.save_qmapper:
    ###########################
    print('### Saving quantile mapping objects ', flush=True)
    ###########################

    # Save trained quantile mapper for experiment
    with open(os.path.join(base_folder, f'fcst_qmapper_{args.num_lat_lon_chunks}.pkl'), 'wb+') as ofh:
        pickle.dump(fcst_qmapper, ofh)
    
    with open(os.path.join(base_folder, f'cgan_qmapper_{args.num_lat_lon_chunks}.pkl'), 'wb+') as ofh:
        pickle.dump(cgan_qmapper, ofh)
        
          
if args.save_data:
    ###########################
    print('### Saving data ', flush=True)
    ###########################
    with open(os.path.join(model_eval_folder, f'fcst_qmap_{args.num_lat_lon_chunks}.pkl'), 'wb+') as ofh:
        print('Fcst corrected shape', quantile_data_dicts['test']['Fcst + qmap'].shape)
        pickle.dump(quantile_data_dicts['test']['Fcst + qmap'], ofh)

    with open(os.path.join(model_eval_folder, f'cgan_qmap_{args.num_lat_lon_chunks}.pkl'), 'wb+') as ofh:
        pickle.dump(cgan_corrected, ofh)

if args.plot:
    ###########################
    print('## Creating plots', flush=True)
    ###########################
    for data_type, quantile_data_dict in tqdm(quantile_data_dicts.items(), file=sys.stdout):
        # Overall q-q plot
        
        fcst_key = 'GAN' if 'GAN' in quantile_data_dict else 'Fcst'
        
        quantile_format_dict = {'GAN': {'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'color': 'k'},
                    'Fcst': {'color': 'r', 'marker': '+', 'alpha': 1},
                    'Fcst + qmap': {'color': 'r', 'marker': 'o', 'alpha': 0.7},
                    'GAN + qmap': {'color': 'b', 'marker': 'o', 'alpha': 0.7}}
        quantile_results, _, _ = get_quantile_data(
            quantile_data_dict=quantile_data_dict,
            range_dict=range_dict,
            save_path=os.path.join(args.output_folder, f'qq_plot_{data_type}_n{args.num_lat_lon_chunks}_total.pdf'))

        plot_quantiles(quantile_results=quantile_results,quantile_data_dict=quantile_data_dict, min_data_points_per_quantile=args.min_points_per_quantile, format_lookup=quantile_format_dict,
                       save_path=os.path.join(args.output_folder, f'qq_plot_{data_type}_n{args.num_lat_lon_chunks}_total.pdf'))

        # Q-Q plot for areas
        fig, ax = plt.subplots(max(2, len(special_areas)),1, figsize=(10, len(special_areas)*10))
        fig.tight_layout(pad=4)
        for n, (area, area_range) in enumerate(special_areas.items()):

            lat_range = area_range['lat_index_range']
            lon_range = area_range['lon_index_range']
            
            local_quantile_data_dict = {}
            for k, v in quantile_data_dict.items():
                local_quantile_data_dict[k] = copy.deepcopy(v)
                local_quantile_data_dict[k] = local_quantile_data_dict[k][:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
            
            try:
                plot_quantiles(local_quantile_data_dict, ax=ax[n], min_data_points_per_quantile=args.min_points_per_quantile,  format_lookup=quantile_format_dict,
                               save_path=os.path.join(args.output_folder, f'qq_plot_{data_type}_n{args.num_lat_lon_chunks}_{area}.pdf'))
                ax[n].set_title(area)
            except:
                print(data_type, ' ', area )
