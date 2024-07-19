import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from argparse import ArgumentParser

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))
from dsrnngan.utils.utils import bootstrap_summary_statistic, get_area_range, special_areas
from dsrnngan.utils import read_config

###########################
# Load model data
###########################

parser = ArgumentParser(description='Script for quantile mapping.')
parser.add_argument('--log-folder', type=str, help='model log folder', required=True)
parser.add_argument('--area', type=str, default='all', choices=list(special_areas.keys()), 
help="Area to run analysis on. Defaults to 'all' which performs analysis over the whole domain")
parser.add_argument('--model-number', type=str, help='model number', required=True)
parser.add_argument('--n-bootstrap-samples', type=int, default=100,
                    help='Number of bootstrap samples to use.')
parser.add_argument('--plot', action='store_true', help='Make plots')
args = parser.parse_args()

# Get lat/lon range from log folder
base_folder = '/'.join(args.log_folder.split('/')[:-1])
data_config = read_config.read_data_config(config_folder=base_folder)
model_config = read_config.read_model_config(config_folder=base_folder)



with open(os.path.join(args.log_folder, f'arrays-{args.model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

n_samples = arrays['truth'].shape[0]
    
truth_array = arrays['truth']
samples_gen_array = arrays['samples_gen']
fcst_array = arrays['fcst_array']
persisted_fcst_array = arrays['persisted_fcst']
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)
dates = [d[0] for d in arrays['dates']]
hours = [h[0] for h in arrays['hours']]

del arrays

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"

if len(samples_gen_array.shape) == 3:
    samples_gen_array = np.expand_dims(samples_gen_array, -1)

(n_samples, width, height, ensemble_size) = samples_gen_array.shape


# Open quantile mapped forecasts
with open(os.path.join(args.log_folder, f'fcst_qmap_3.pkl'), 'rb') as ifh:
    fcst_corrected = pickle.load(ifh)

with open(os.path.join(args.log_folder, f'cgan_qmap_2.pkl'), 'rb') as ifh:
    cgan_corrected = pickle.load(ifh)
    if len(cgan_corrected.shape) == 3:
        cgan_corrected = np.expand_dims(cgan_corrected, -1)
    
###########################

###########################

# Filter by area

# Locations
latitude_range, lat_range_index, longitude_range, lon_range_index = get_area_range(data_config, area=args.area)

cgan_corrected = cgan_corrected[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_corrected = fcst_corrected[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]

truth_array = truth_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
samples_gen_array = samples_gen_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_array = fcst_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1 ]
ensmean_array = np.mean(samples_gen_array, axis=-1)[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]

  
    
# calculate standard deviation and mean of
quantile_locations = [np.round(1 - 10**(-n),n+1) for n in range(3, 9)]

def calculate_quantiles(input_array, quantile_locations=quantile_locations):
    
    return np.quantile(input_array, quantile_locations)

fcst_quantiles = calculate_quantiles(fcst_corrected)
cgan_quantiles = calculate_quantiles(cgan_corrected)
obs_quantiles = calculate_quantiles(truth_array)


bootstrap_results_dict_obs = bootstrap_summary_statistic(calculate_quantiles, truth_array, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
bootstrap_results_dict_fcst_qmap = bootstrap_summary_statistic(calculate_quantiles, fcst_corrected, n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
bootstrap_results_dict_cgan_qmap = bootstrap_summary_statistic(calculate_quantiles, cgan_corrected[...,0], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)

# Save results 
with open(os.path.join(args.log_folder, f'bootstrap_quantile_results_n{args.n_bootstrap_samples}_{args.area}.pkl'), 'wb+') as ofh:
    pickle.dump({'obs': bootstrap_results_dict_obs, 
                 'cgan': bootstrap_results_dict_cgan_qmap, 
                 'fcst': bootstrap_results_dict_fcst_qmap}, ofh)

if args.plot:
    fig, ax = plt.subplots(1,1)

    ax.errorbar(obs_quantiles, fcst_quantiles, yerr=2*bootstrap_results_dict_fcst_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
    ax.plot(obs_quantiles, obs_quantiles, '--')
    ax.set_xlabel('Model (mm/hr)')

    ax.set_ylabel('Observations (mm/hr)')
    plt.savefig('quantile_bootstrap_intervals_fcst_total.pdf', format='pdf')

    fig, ax = plt.subplots(1,1)

    ax.errorbar(obs_quantiles, cgan_quantiles, yerr=2*bootstrap_results_dict_cgan_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
    ax.plot(obs_quantiles, obs_quantiles, '--')
    ax.set_xlabel('Model (mm/hr)')

    ax.set_ylabel('Observations (mm/hr)')
    plt.savefig('quantile_bootstrap_intervals_cgan_total.pdf', format='pdf')



    # Same for all the areas
    quantile_locations =  [np.round(1 - 10**(-n),n+1) for n in range(1, 7)]
    n_cols = 2

    fig, ax = plt.subplots(int(np.round(len(special_areas)/2)),2, figsize=(2*3, int(np.round(len(special_areas)/2))*3))
    fig.tight_layout(pad=4)
    for n, (area, area_range) in enumerate(special_areas.items()):
        row = int(n/2)
        column = n %2
        print(area)
        lat_range = area_range['lat_index_range']
        lon_range = area_range['lon_index_range']
        
        cgan_quantiles = calculate_quantiles(cgan_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)
        obs_quantiles = calculate_quantiles(truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)

        bootstrap_results_dict_obs = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), truth_array[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_dict_cgan_qmap = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), cgan_corrected[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        
        ax[row, column].errorbar(obs_quantiles, cgan_quantiles, yerr=2*bootstrap_results_dict_cgan_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
        ax[row, column].plot(obs_quantiles, obs_quantiles, '--')
        ax[row, column].set_xlabel('Observations (mm/hr)')
    #     ax[row, column].set_xticks(obs_quantiles)
    #     ax[row, column].set_xticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])

        ax[row, column].set_ylabel('Model (mm/hr)')

    #     ax[row, column].set_yticks(cgan_quantiles)
    #     ax[row, column].set_yticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])
        ax[row, column].set_title(area)
    fig.tight_layout(pad=2.0)
    plt.savefig('quantile_bootstrap_intervals_cgan_area.pdf', format='pdf')


    # Same for all the areas
    quantile_locations =  [np.round(1 - 10**(-n),n+1) for n in range(1, 7)]
    n_cols = 2

    fig, ax = plt.subplots(int(np.round(len(special_areas)/2)),2, figsize=(2*3, int(np.round(len(special_areas)/2))*3))
    fig.tight_layout(pad=4)
    for n, (area, area_range) in enumerate(special_areas.items()):
        row = int(n/2)
        column = n %2
        print(area)
        lat_range = area_range['lat_index_range']
        lon_range = area_range['lon_index_range']
        
        fcst_quantiles = calculate_quantiles(fcst_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)
        obs_quantiles = calculate_quantiles(truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], quantile_locations=quantile_locations)

        bootstrap_results_dict_obs = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), truth_array[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        bootstrap_results_dict_fcst_qmap = bootstrap_summary_statistic(lambda x: calculate_quantiles(x, quantile_locations), fcst_corrected[0::2, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]], n_bootstrap_samples=args.n_bootstrap_samples, time_resample=True)
        
        ax[row, column].errorbar(obs_quantiles, fcst_quantiles, yerr=2*bootstrap_results_dict_fcst_qmap['std'], xerr=2*bootstrap_results_dict_obs['std'], capsize=2)
        ax[row, column].plot(obs_quantiles, obs_quantiles, '--')
        ax[row, column].set_xlabel('Observations (mm/hr)')
    #     ax[row, column].set_xticks(obs_quantiles)
    #     ax[row, column].set_xticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])

        ax[row, column].set_ylabel('Model (mm/hr)')

    #     ax[row, column].set_yticks(cgan_quantiles)
    #     ax[row, column].set_yticklabels([f'$10^{{ {int(np.round(np.log10(1 -q),0))}}}$' for q in quantile_locations])
        ax[row, column].set_title(area)
    fig.tight_layout(pad=2.0)
    plt.savefig('quantile_bootstrap_intervals_fcst_area.pdf', format='pdf')