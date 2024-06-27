import pickle
import os, sys
import copy
from tqdm import tqdm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import tempfile
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
from matplotlib import gridspec
from metpy import plots as metpy_plots
from matplotlib.colors import ListedColormap, BoundaryNorm
from properscoring import crps_ensemble
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from string import ascii_lowercase

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.utils import read_config
from dsrnngan.utils.utils import load_yaml_file, get_best_model_number, special_areas, get_area_range
from dsrnngan.evaluation.plots import plot_contourf, range_dict, quantile_locs, percentiles, plot_quantiles, get_quantile_data, border_feature, disputed_border_feature
from dsrnngan.data.data import denormalise, DEFAULT_LATITUDE_RANGE, DEFAULT_LONGITUDE_RANGE
from dsrnngan.data import data
from dsrnngan.evaluation.rapsd import  rapsd
from dsrnngan.evaluation.scoring import fss, get_spread_error_data, get_metric_by_hour
from dsrnngan.evaluation.evaluation import get_diurnal_cycle, get_fss_scores
from dsrnngan.evaluation.benchmarks import QuantileMapper, empirical_quantile_map, quantile_map_grid

def clip_outliers(data, lower_pc=2.5, upper_pc=97.5):
    
    data_clipped = copy.deepcopy(data)
    data_clipped[data_clipped<np.percentile(data_clipped, lower_pc)] = np.percentile(data_clipped, lower_pc)  #using percentiles rather than indexing a sorted list of the array values allows this to work even when data is a small array. (I've not checked if this works for masked data.)
    data_clipped[data_clipped>np.percentile(data_clipped, upper_pc)] = np.percentile(data_clipped, upper_pc)

    return data_clipped

# This dict chooses which plots to create

plot_persistence = False

format_lookup = {'GAN': {'color': 'b'}, 
                 'IFS': {'color': 'r'}, 
                 'GAN + qmap': {'color': 'b', 'linestyle': '--'}, 
                 'IFS + qmap': {'color': 'r', 'linestyle': '--'},
                 'Obs (IMERG)': {'color': 'k', 'linestyle': '-'}}

################################################################################
## Setup
################################################################################

parser = ArgumentParser(description='Plotting script.')

parser.add_argument('--output-dir', type=str, required=True, help="Folder to store the plots in")
parser.add_argument('--nickname', type=str, required=True, help="nickname to give this model")
parser.add_argument('--model-eval-folder', type=str, required=True, help="Folder containing pre-evaluated cGAN data")
parser.add_argument('--model-number', type=int, required=True, help="Checkpoint number of model")
parser.add_argument('--area', type=str, default='all', choices=list(special_areas.keys()), 
help="Area to run analysis on. Defaults to 'All' which performs analysis over the whole domain")
parser.add_argument('--num-samples', type=int, default=None, help="Number of samples to use")
parser.add_argument('--climatological-data-path', type=str, required=True, help="Folder containing climatological data to load")
metric_group = parser.add_argument_group('metrics')
metric_group.add_argument('-ex', '--examples', action="store_true", help="Plot a selection of example precipitation forecasts")
metric_group.add_argument('-sc', '--scatter', action="store_true", help="Plot scatter plots of domain averaged rainfall")
metric_group.add_argument('-rh', '--rank-hist', action="store_true", help="Plot rank histograms")
metric_group.add_argument('-rmse', action="store_true", help="Plot root mean square error")
metric_group.add_argument('-bias', action="store_true", help="Plot bias")
metric_group.add_argument('-se', '--spread-error', action="store_true", help="Plot the spread error")
metric_group.add_argument('-rapsd', action="store_true", help="Plot the radially averaged power spectral density")
metric_group.add_argument('-qq', '--quantiles', action="store_true", help="Create quantile-quantile plot")
metric_group.add_argument('-hist', action="store_true", help="Plot Histogram of rainfall intensities")
metric_group.add_argument('-crps', action="store_true", help="Plot CRPS scores")
metric_group.add_argument('-fss', action="store_true", help="PLot fractions skill score")
metric_group.add_argument('-d', '--diurnal', action="store_true", help="Plot diurnal cycle")
metric_group.add_argument('-conf', '--confusion-matrix', action="store_true", help="Calculate confusion matrices")
metric_group.add_argument('-csi', action="store_true", help="Plot CSI and ETS")
parser.add_argument('--debug', action='store_true', help="Debug flag to use small amounts of data")

args = parser.parse_args()

nickname = args.nickname
area = args.area

all_metrics = ['examples', 'scatter','rank_hist','rmse', 'bias', 'spread_error','rapsd','quantiles','hist','crps','fss','diurnal','confusion_matrix','csi']
metric_dict = {metric_name: args.__getattribute__(metric_name) for metric_name in all_metrics}


# If in debug mode, then don't overwrite existing plots
if args.debug:
    temp_stats_dir = tempfile.TemporaryDirectory()
    args.output_dir = temp_stats_dir.name

model_number = args.model_number
model_eval_folder = args.model_eval_folder

folder_suffix = model_eval_folder.split('/')[-1]
args.output_dir = os.path.join(args.output_dir, folder_suffix)
os.makedirs(args.output_dir, exist_ok=True)

# Get lat/lon range from log folder
base_folder = '/'.join(model_eval_folder.split('/')[:-1])
try:
    config = load_yaml_file(os.path.join(base_folder, 'setup_params.yaml'))
    model_config, data_config = read_config.get_config_objects(config)
except FileNotFoundError:
    data_config = read_config.read_data_config(config_folder=base_folder)

# Locations
latitude_range, lat_range_index, longitude_range, lon_range_index = get_area_range(data_config, area=args.area)

###################################
# Load arrays
##################################

with open(os.path.join(model_eval_folder, f'arrays-{model_number}.pkl'), 'rb') as ifh:
    arrays = pickle.load(ifh)

if args.debug:
    n_samples = 100
elif args.num_samples:
    n_samples = min(args.num_samples, arrays['truth'].shape[0])
else:
    n_samples = arrays['truth'].shape[0]


truth_array = arrays['truth'][:n_samples, ::]
samples_gen_array = arrays['samples_gen'][:n_samples, ::]
fcst_array = arrays['fcst_array'][:n_samples, ::]
ensmean_array = np.mean(arrays['samples_gen'], axis=-1)[:n_samples, ::]
dates = [d[0] for d in arrays['dates']][:n_samples]
hours = [h[0] for h in arrays['hours']][:n_samples]

del arrays

# Times in EAT timezone
eat_datetimes = [datetime(d.year, d.month, d.day, hours[n]).replace(tzinfo=timezone.utc).astimezone(ZoneInfo('Africa/Nairobi')) for n,d in enumerate(dates)]
dates = [datetime(d.year, d.month, d.day) for d in eat_datetimes]
hours = [d.hour for d in eat_datetimes]

assert len(set(list(zip(dates, hours)))) == fcst_array.shape[0], "Degenerate date/hour combinations"
(n_samples, width, height, ensemble_size) = samples_gen_array.shape

# Find dry and rainy days in sampled dataset
means = [(n, truth_array[n,:,:].mean()) for n in range(n_samples)]
sorted_means = sorted(means, key=lambda x: x[1])

n_extreme_days = 10
wet_day_indexes = [item[0] for item in sorted_means[-10:]]
dry_day_indexes = [item[0] for item in sorted_means[:10]]


        
####################################
### Load in quantile-mapped data (created by scripts/quantile_mapping.py)
####################################

if os.path.isfile(os.path.join(model_eval_folder, 'fcst_qmap_3.pkl')):
    with open(os.path.join(model_eval_folder, 'fcst_qmap_3.pkl'), 'rb') as ifh:
        fcst_corrected = pickle.load(ifh)
else:
    with open(os.path.join(base_folder, 'fcst_qmapper_3.pkl'), 'rb') as ifh:
        fcst_qmapper = pickle.load(ifh) 
    print('Performing forecast quantile mapping', flush=True)
    fcst_corrected = fcst_qmapper.get_quantile_mapped_forecast(fcst=fcst_array, dates=dates, hours=hours)
    print('Finished forecast quantile mapping', flush=True)
    
    with open(os.path.join(model_eval_folder, 'fcst_qmap_3.pkl'), 'wb+') as ofh:
        pickle.dump(fcst_corrected, ofh)

if os.path.isfile(os.path.join(model_eval_folder, 'cgan_qmap_2.pkl')):
    with open(os.path.join(model_eval_folder, 'cgan_qmap_2.pkl'), 'rb') as ifh:
        cgan_corrected = pickle.load(ifh)
else:
    with open(os.path.join(base_folder, 'cgan_qmapper_2.pkl'), 'rb') as ifh:
        cgan_qmapper = pickle.load(ifh)
    print('Performing cGAN quantile mapping', flush=True)

    cgan_corrected = np.empty(shape=samples_gen_array.shape)
    cgan_corrected[...] = np.nan
    for en in tqdm(range(ensemble_size)):
        print(en, flush=True)

        cgan_corrected[:,:,:,en] = cgan_qmapper.get_quantile_mapped_forecast(fcst=samples_gen_array[:,:,:,en], dates=dates, hours=hours)
    print('Finished cgan quantile mapping', flush=True)
    
    with open(os.path.join(model_eval_folder, 'cgan_qmap_2.pkl'), 'wb+') as ofh:
        pickle.dump(cgan_corrected, ofh)

cgan_corrected = cgan_corrected[:n_samples, ::]
fcst_corrected = fcst_corrected[:n_samples, ::]

print('shape of cgan corrected: ', cgan_corrected.shape, flush=True)

# Filter by area
cgan_corrected = cgan_corrected[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_corrected = fcst_corrected[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]

truth_array = truth_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]
samples_gen_array = samples_gen_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1,:]
fcst_array = fcst_array[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1 ]
ensmean_array = np.mean(samples_gen_array, axis=-1)[:, lat_range_index[0]:lat_range_index[1]+1, lon_range_index[0]:lon_range_index[1]+1]



# Only keep one ensemble member if we aren't looking at distribution based things
if not args.rank_hist and not args.spread_error and not args.crps and not args.examples and samples_gen_array.shape[-1] > 1:
    samples_gen_array = samples_gen_array[...,:1]
    cgan_corrected = cgan_corrected[...,:-1]
    
data_dict = {
                        'GAN': samples_gen_array[:, :, :, 0],
                        'Obs (IMERG)': truth_array,
                        'IFS': fcst_array,
                        'IFS + qmap': fcst_corrected,
                        'GAN + qmap': cgan_corrected[:,:,:,0]}
          
################################################################################
## Climatological data for comparison.
print('Loading climate data',flush=True)

all_imerg_data = []
all_ifs_data = []

if args.debug:
    year_range = [2004]
    month_range = [1]
else:
    year_range = range(2003, 2018)
    month_range = range(1,13)

for year in tqdm(year_range):
    for month in month_range:

        imerg_ds = xr.open_dataarray(os.path.join(args.climatological_data_path, f'daily_imerg_rainfall_{month}_{year}.nc'))

        imerg_data = imerg_ds.sel(lat=latitude_range, method='nearest').sel(lon=longitude_range, method='nearest').values

        for t in range(imerg_data.shape[0]):
            
            snapshot = imerg_data[t, :, :]
            
            all_imerg_data.append(snapshot)
            
all_imerg_data = np.stack(all_imerg_data, axis = 0)

daily_historical_avg = np.mean(all_imerg_data, axis=0)
daily_historical_std = np.std(all_imerg_data, axis=0)

hourly_historical_avg = daily_historical_avg / 24
hourly_historical_std = daily_historical_std / 24

#################################################################################################
## Plot examples
#################################################################################################


if metric_dict['examples']:
    print('*********** Plotting Examples **********************')
    tp_index = data.all_ifs_fields.index('tp')

    # plot configurations
    levels = [0, 0.1, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150] # in units of log10
    precip_cmap = ListedColormap(metpy_plots.ctables.colortables["precipitation"][:len(levels)-1], 'precipitation')
    precip_norm = BoundaryNorm(levels, precip_cmap.N)
    plt.rcParams.update({'font.size': 11})
    spacing = 10
    units = "Rain rate [mm h$^{-1}$]"
    precip_levels=np.arange(0, 1.5, 0.15)

    indexes = wet_day_indexes[:4] + dry_day_indexes[:4]

    num_cols = 6
    num_samples = len(indexes)
    num_rows = num_samples

    rows = [[f'cgan_sample_{n}', f'cgan_mean_{n}', f'imerg_{n}', f'ifs_{n}', f'ifs_qmap_{n}', 'cbar'] for n in range(num_rows)]

    fig = plt.figure(constrained_layout=True, figsize=(2.5*num_cols, 3*num_rows))
    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]*(num_cols - 1) + [0.05],
                        height_ratios=[1]*(num_rows) + [0.05],
                        wspace=0.005)                      
    for n in tqdm(range(len(indexes))):

        ix = indexes[n]
        img_gens = samples_gen_array[ix, :,:,:]
        truth = truth_array[ix,:,:]
        fcst = fcst_array[ix,:,:]
        fcst_corr = fcst_corrected[ix, :, :]
        date = dates[ix]
        hour = hours[ix]
        avg_img_gens = img_gens.mean(axis=-1)
        date_str = date.strftime('%d-%m-%Y') + f' {hour:02d}:00:00'
        
        # cGAN
        data_lookup = {'cgan_sample': {'data': img_gens[:,:,0], 'title': 'cGAN sample'},
                    'cgan_mean': {'data': avg_img_gens, 'title': f'cGAN sample average'},
                    'imerg' : {'title': f"IMERG: {date_str}", 'data': truth},
                    'ifs': {'data': fcst, 'title': 'IFS'},
                    'ifs_qmap': {'data': fcst_corr, 'title': 'IFS qmap'}
                    }
        for col, (k, val) in enumerate(data_lookup.items()):
    
            ax = fig.add_subplot(gs[n, col], projection = ccrs.PlateCarree())
            ax.coastlines(resolution='10m', color='black', linewidth=0.4)
            
            clipped_data = clip_outliers(val['data'], lower_pc=0.1, upper_pc=99.9)
            im = ax.imshow(val['data'],
                    interpolation='nearest',
                    norm=precip_norm,
                    cmap=precip_cmap,
                    origin='lower',
                    extent=[min(longitude_range), max(longitude_range), 
                    min(latitude_range), max(latitude_range)],
                    transform=ccrs.PlateCarree(),
                    alpha=0.8)
            ax.add_feature(border_feature)
            ax.add_feature(disputed_border_feature)
            ax.set_title(val['title'])
            
    cbar_ax = fig.add_subplot(gs[-1, :])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10,
                    )
    cb.ax.set_xlabel("Precipitation (mm / hr)", loc='center')


    plt.savefig(os.path.join(args.output_dir, f'cGAN_samples_IFS_{nickname}_{model_number}_{area}.pdf'), format='pdf')

##################################################################################
## Binned scatter plot
##################################################################################

if metric_dict['scatter']:
    from scipy.stats import pearsonr

    metric_types = {'mean': lambda x: np.mean(x)}
    metric_name = 'mean'

    scatter_results = {}
    for metric_name, metric_fn in tqdm(metric_types.items()):
        if metric_name == 'mean':

            scatter_results[metric_name] = {}

            scatter_results[metric_name]['cGAN-qm'] = [metric_fn(cgan_corrected[n,:,:,0]) for n in range(cgan_corrected.shape[0])]
            scatter_results[metric_name]['IMERG'] = [metric_fn(truth_array[n,:,:]) for n in range(truth_array.shape[0])]
            scatter_results[metric_name]['IFS-qm'] = [metric_fn(fcst_corrected[n,:,:]) for n in range(fcst_corrected.shape[0])]

    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].scatter(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['cGAN-qm'] , marker='+', label='cGAN-qm')
    ax[1].scatter(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['IFS-qm'], marker='+',label='IFS-qm')
    max_val = np.max(scatter_results[metric_name]['cGAN-qm']  + scatter_results[metric_name]['IMERG'] + scatter_results[metric_name]['IFS-qm'])
    for a in ax:
        a.plot(scatter_results[metric_name]['IMERG'], scatter_results[metric_name]['IMERG'], 'k--')
        a.legend(loc='lower right')
        a.set_xlabel('IMERG observations (mm/hr)')
        a.set_ylabel('Forecasts (mm/hr)')
        a.set_xlim([0, max_val])
        a.set_ylim([0, max_val])
    print(metric_name)
    plt.savefig(os.path.join(args.output_dir, f'scatter_{nickname}_{model_number}_{area}.pdf'), format='pdf')


##################################################################################
###  Rank histogram
##################################################################################



if metric_dict['rank_hist']:
    from pysteps.verification.ensscores import rankhist
    print('*********** Plotting Rank histogram **********************')
    rank_cgan_array = np.moveaxis(cgan_corrected, -1, 0)
    num_ensemble_members = 100
    n_samples = 500
    ranks = rankhist(rank_cgan_array[:num_ensemble_members,:n_samples,...], truth_array[:n_samples,...], normalize=True)
    
    # Rank histogram above threshold
    thr = 0.1
    rank_cgan_array_above_thr = []
    for en in range(cgan_corrected.shape[-1]):
        rank_cgan_array_above_thr.append(cgan_corrected[:n_samples,:,:,en][ensmean_array[:n_samples,:,:] > thr])
    rank_cgan_array_above_thr = np.stack(rank_cgan_array_above_thr, axis=0)
    rank_cgan_array_above_thr = np.expand_dims(rank_cgan_array_above_thr, -1)
    
    rank_obs_array_above_thr = np.expand_dims(truth_array[:n_samples,:,:][ensmean_array[:n_samples,:,:] > thr], axis=-1)
    
    ranks_above_thr = rankhist(rank_cgan_array_above_thr, rank_obs_array_above_thr, normalize=True)


    # Rank histogram below threshold
    rank_cgan_array_below_thr = []
    for en in range(cgan_corrected.shape[-1]):
        rank_cgan_array_below_thr.append(cgan_corrected[:n_samples,:,:,en][ensmean_array[:n_samples,:,:] <= thr])
    rank_cgan_array_below_thr = np.stack(rank_cgan_array_below_thr, axis=0)
    rank_cgan_array_below_thr = np.expand_dims(rank_cgan_array_below_thr, -1)

    rank_obs_array_below_thr = np.expand_dims(truth_array[:n_samples,:,:][ensmean_array[:n_samples,:,:] <= thr], axis=-1)

    ranks_below_thr = rankhist(rank_cgan_array_below_thr, rank_obs_array_below_thr, normalize=True)
    
    with open(os.path.join(args.output_dir, f'ranks_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(ranks, ofh)
        
    with open(os.path.join(args.output_dir, f'ranks_above_thr_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(ranks_above_thr, ofh)
    
    with open(os.path.join(args.output_dir, f'ranks_below_thr_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(ranks_below_thr, ofh)
            
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.bar(np.linspace(0,1,num_ensemble_members+1), ranks, width=1/num_ensemble_members, 
        color='cadetblue', edgecolor='grey')
    ax.set_ylim([0,0.08])
    ax.set_xlim([0-0.5/num_ensemble_members,1+0.5/num_ensemble_members])

    ax.hlines(1/num_ensemble_members, 0-0.5/num_ensemble_members,1+0.5/num_ensemble_members, linestyles='dashed', colors=['k'])
    ax.set_xlabel('Normalised rank')
    ax.set_ylabel('Normalised frequency')
    plt.savefig(os.path.join(args.output_dir,f'rank_hist_{nickname}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')
    
#################################################################################
## Spread error

if metric_dict.get('spread_error') and ensemble_size > 2:
    print('*********** Plotting spread error **********************')
    plt.rcParams.update({'font.size': 20})
    
    upper_percentile = 99
    num_bins=100
    
    fig, ax = plt.subplots(1,1)
    binned_mse, binned_spread = get_spread_error_data(n_samples=4000, observation_array=truth_array, 
                                                      ensemble_array=cgan_corrected, 
                                                      n_bins=100)
    if np.isnan(binned_spread[0]):
        binned_spread = binned_spread[1:]
        binned_mse = binned_mse[1:]
        
    ax.plot(binned_spread, binned_mse, marker='+', markersize=10)
    ax.plot(np.linspace(0,max(binned_spread),10), np.linspace(0,max(binned_spread),10), 'k--')
    ax.set_ylabel('MSE (mm/hr)')
    ax.set_xlabel('Ensemble spread (mm/hr)')
    ax.set_title(f'Spread-error')
    plt.savefig(os.path.join(args.output_dir,f'spread_error_{nickname}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')

    with open(os.path.join(args.output_dir, f'spread_error_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump({'binned_spread': binned_spread, 'binned_mse': binned_mse}, ofh)
#################################################################################
## RAPSD
plt.rcParams.update({'font.size': 16})
if metric_dict['rapsd']:
    rapsd_data_dict = {
                        'GAN': cgan_corrected[:, :, :, 0],
                        'Obs (IMERG)': truth_array,
                        'IFS': fcst_corrected}
    
    print('*********** Plotting RAPSD **********************')
    rapsd_results = {}
    for k, v in rapsd_data_dict.items():
            rapsd_results[k] = []
            for n in tqdm(range(n_samples)):
                
                fft_freq_pred = rapsd(v[n,:,:], fft_method=np.fft)
                rapsd_results[k].append(fft_freq_pred)
        

            rapsd_results[k] = np.mean(np.stack(rapsd_results[k], axis=-1), axis=-1)
    
    # Save results
    with open(os.path.join(args.output_dir,f'rapsd_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(rapsd_results, ofh)

    # plot
    fig, ax = plt.subplots(1,1)

    for k, v in rapsd_results.items():
        ax.plot(v, label=k, **format_lookup[k])
    
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylabel('Power Spectral Density')
    ax.set_xlabel('Wavenumber')
    ax.legend()
    
    plt.savefig(os.path.join(args.output_dir,f'rapsd_{nickname}_{model_number}_{area}.pdf'), format='pdf', bbox_inches='tight')

#################################################################################
## Q-Q plot


if metric_dict['quantiles']:

    print('*********** Plotting Q-Q  **********************')
    quantile_format_dict = {'GAN': {'color': 'b', 'marker': '+', 'alpha': 1},
                    'Obs (IMERG)': {'color': 'k'},
                    'IFS': {'color': 'r', 'marker': '+', 'alpha': 1},
                    'IFS + qmap': {'color': 'r', 'marker': 'o', 'alpha': 0.7},
                    'GAN + qmap': {'color': 'b', 'marker': 'o', 'alpha': 0.7}}
    quantile_data_dict = {
                        'GAN': samples_gen_array[:, :, :, 0],
                        'Obs (IMERG)': truth_array,
                        'IFS': fcst_array,
                        'IFS + qmap':fcst_corrected,
                        'GAN + qmap': cgan_corrected[:,:,:,0]}
    quantile_results, quantile_boundaries, intervals = get_quantile_data(quantile_data_dict)
    fig, ax = plt.subplots(1,1)
    plot_quantiles(quantile_results=quantile_results, 
                   quantile_data_dict=quantile_data_dict, 
                   format_lookup=quantile_format_dict, ax=ax,
                   save_path=os.path.join(args.output_dir,f'quantiles_total_{nickname}_{model_number}_{area}.pdf'))

    # Quantiles for different areas

    # percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
    # percentiles=np.concatenate(percentiles_list)
    # quantile_boundaries = [item / 100 for item in percentiles]

    # fig, ax = plt.subplots(max(2, len(special_areas)),1, figsize=(10, len(special_areas)*10))
    # fig.tight_layout(pad=4)
    # for n, (area, area_range) in enumerate(special_areas.items()):

    #     lat_range = area_range['lat_index_range']
    #     lon_range = area_range['lon_index_range']
        
    #     quantile_data_dict = {
    #                 'GAN': samples_gen_array[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1], 0],
    #                 'Obs (IMERG)': truth_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]],
    #                 'IFS': fcst_array[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]],
    #                 'IFS + qmap': fcst_corrected[:,lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]],
    #                 'GAN + qmap': cgan_corrected[:, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1], 0]}
               
    #     _, ax[n] = plot_quantiles(quantile_data_dict=quantile_data_dict, format_lookup=quantile_format_dict, ax=ax[n], min_data_points_per_quantile=1)

        
    # fig.tight_layout(pad=2.0)
    # plt.savefig(os.path.join(args.output_dir,f'quantiles_area_{nickname}_{model_number}.pdf'), format='pdf')



#################################################################################
## Histograms



if metric_dict['hist']:
    print('*********** Plotting Histograms **********************')
    
    (q_99pt9, q_99pt99) = np.quantile(truth_array, [0.999, 0.9999])


    fig, axs = plt.subplots(2,1, figsize=(10,10))
    fig.tight_layout(pad=4)
    bin_boundaries=np.arange(0,300,4)

    data_dict = {'Obs (IMERG)': {'data': truth_array, 'histtype': 'stepfilled', 'alpha':0.6, 'facecolor': 'grey'}, 
                'IFS': {'data': fcst_array, 'histtype': 'step', 'edgecolor': 'red'},
                'IFS + qmap': {'data': fcst_corrected, 'histtype': 'step', 'edgecolor': 'red', 'linestyle': '--'},
                'cGAN': {'data': samples_gen_array[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue'},
                'cGAN + qmap': {'data': cgan_corrected[:,:,:,0], 'histtype': 'step', 'edgecolor': 'blue', 'linestyle': '--'}}
    rainfall_amounts = {}

    edge_colours = ["blue", "green", "red", 'orange']
    for n, (name, d) in enumerate(data_dict.items()):
        
        axs[0].hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, alpha=d.get('alpha'),
                    facecolor=d.get('facecolor'), edgecolor=d.get('edgecolor'), linestyle= d.get('linestyle'))
        
        axs[1].hist(d['data'].flatten(), bins=bin_boundaries, histtype=d['histtype'], label=name, weights=d['data'].flatten(), alpha=d.get('alpha'),
                    facecolor=d.get('facecolor'), edgecolor=d.get('edgecolor'), linestyle= d.get('linestyle'))

        
    for ax in axs:
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Average hourly rainfall in bin (mm/hr)')
        ax.vlines(q_99pt99, 0, 10**8, linestyles='--')
        ax.text(q_99pt99 + 7 , 10**7, '$99.99^{th}$')
    plt.savefig(os.path.join(args.output_dir,f'histograms_{nickname}_{model_number}_{area}.pdf'), format='pdf')

if metric_dict.get('rmse'):
    #################################################################################
    ## Bias and RMSE
    # RMSE
    rmse_dict = {'GAN': np.sqrt(np.mean(np.square(truth_array - samples_gen_array[:,:,:,0]), axis=0)),
            'GAN + qmap' : np.sqrt(np.mean(np.square(truth_array - cgan_corrected[:n_samples,:,:,0]), axis=0)),
            'IFS' : np.sqrt(np.mean(np.square(truth_array - fcst_array), axis=0)),
            'IFS + qmap' : np.sqrt(np.mean(np.square(truth_array - fcst_corrected[:n_samples,:,:]), axis=0))}



    fig, ax = plt.subplots(len(rmse_dict.keys())+1,2, 
                        subplot_kw={'projection' : ccrs.PlateCarree()},
                        figsize=(12,16))


    max_level = 10
    value_range = list(np.arange(0, max_level, max_level / 50))

    value_range_2 = list(np.arange(0, 5, 5 / 50))

    for n, (k, v) in enumerate(rmse_dict.items()):

        im = plot_contourf(ax=ax[n,0], data=v, title=k, value_range=value_range, lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im, ax=ax[n,0])
        ax[n,0].set_title(k)
        im2 = plot_contourf(ax=ax[n,1], data=v / hourly_historical_std, title=k + ' / truth_std', value_range=value_range_2 , lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im2, ax=ax[n,1])

    im = plot_contourf(ax=ax[n+1,0], data=hourly_historical_std, title='truth_std', lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+1,0])

    plt.savefig(os.path.join(args.output_dir,f'rmse_{nickname}_{model_number}_{area}.pdf'), format='pdf')
    

if metric_dict['bias']:
    bias_dict = {'GAN': np.mean(samples_gen_array[:,:,:,0] - truth_array, axis=0),
        'GAN + qmap' : np.mean(cgan_corrected[:n_samples,:,:,0] - truth_array, axis=0),
        'IFS' : np.mean(fcst_array - truth_array, axis=0),
        'IFS + qmap' : np.mean(fcst_corrected[:n_samples, :,:] - truth_array, axis=0)}
    
    # Bias

    fig, ax = plt.subplots(len(bias_dict.keys())+2,2, 
                        subplot_kw={'projection' : ccrs.PlateCarree()},
                        figsize=(12,16))

    max_bias_val = max([v.max() for v in bias_dict.values()])

    for n, (k,v) in enumerate(bias_dict.items()):


        # value_range = list(np.arange(-0.5, 0.5, 0.01))
        value_range = None

        im = plot_contourf(ax=ax[n,0], data=v, title=k, cmap='RdBu', value_range=list(np.arange(-1.5, 1.5, 3 / 50)), lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im, ax=ax[n,0])
        
        im = plot_contourf(ax=ax[n,1], data=100* v / hourly_historical_avg, title=f'100 * {k} / truth_avg', value_range=list(np.arange(-200, 200, 400 / 50)),
                        cmap='RdBu', lat_range=latitude_range, lon_range=longitude_range)
        plt.colorbar(im, ax=ax[n,1])
        
    im = plot_contourf(ax=ax[n+1,0], data=hourly_historical_avg, title='truth_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+1,0])

    im = plot_contourf(ax=ax[n+1,1], data=np.mean(np.mean(samples_gen_array, axis=-1), axis=0), title='samples_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+1,1])

    im = plot_contourf(ax=ax[n+2,0], data=np.mean(fcst_array, axis=0), title='fcst_avg', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+2,0])

    im = plot_contourf(ax=ax[n+2,1], data=np.mean(fcst_array, axis=0), title='', cmap='Reds',
                    value_range=list(np.arange(0, 0.6, 0.01)), lat_range=latitude_range, lon_range=longitude_range)
    plt.colorbar(im, ax=ax[n+2,1])

    plt.savefig(os.path.join(args.output_dir,f'bias_{nickname}_{model_number}_{area}.pdf'), format='pdf')
    
    # RMSE by hour
    from dsrnngan.evaluation.scoring import mse, get_metric_by_hour

    fig, ax = plt.subplots(1,1)
    data_dict = {'Obs (IMERG)': truth_array,
                        'GAN + qmap': cgan_corrected[:,:,:,0],
                        'IFS + qmap': fcst_corrected}

    for name, arr in data_dict.items():
        if name != 'Obs (IMERG)':
            metric_by_hour, hour_bin_edges = get_metric_by_hour(mse, obs_array=truth_array, fcst_array=arr, hours=hours, bin_width=3)
            ax.plot(metric_by_hour.keys(), metric_by_hour.values(), label=name)
            ax.set_xticks(np.array(list(metric_by_hour.keys())) - .5)
            ax.set_xticklabels(hour_bin_edges)
    ax.legend()
    plt.savefig(os.path.join(args.output_dir,f'rmse_hours_{nickname}_{model_number}_{area}.pdf'), format='pdf')

#################################################################################

## CRPS

if metric_dict['crps']:
    
    print('*********** Plotting CRPS **********************')
    
    # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
    crps_score_grid = crps_ensemble(truth_array, samples_gen_array)
    crps_score = crps_score_grid.mean()


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection' : ccrs.PlateCarree()}, figsize=(5,5))

    max_level = 10
    value_range = list(np.arange(0, max_level, max_level / 50))

    im = ax.contourf(longitude_range, latitude_range, np.mean(crps_score_grid, axis=0), transform=ccrs.PlateCarree(),
                        cmap='Reds', 
                        # levels=value_range, norm=colors.Normalize(min(value_range), max(value_range)),
                        extend='both')

    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(args.output_dir,f'crps_{nickname}_{model_number}_{area}.pdf'), format='pdf')


#################################################################################

## Fractions Skill Score

if metric_dict['fss']:
    print('*********** Plotting FSS **********************')
    from dsrnngan.evaluation.evaluation import get_fss_scores
    from dsrnngan.evaluation.plots import plot_fss_scores

    # bootstrapping results (previously calculated using scripts/bootstrap_fss.py)
    with open(os.path.join(model_eval_folder, f'bootstrap_fss_results_n50_{area}.pkl'), 'rb') as ifh:
        fss_bstrap_results = pickle.load(ifh)

    quantile_locs = list(fss_bstrap_results['cgan'].keys())
    window_sizes = list(fss_bstrap_results['cgan'][0.9].keys())


    scales = [2*w+1 for w in window_sizes]
    n_samples = -1
    fss_data_dict = {
                        'cGAN': cgan_corrected[:n_samples, :, :, 0],
                        'IFS': fcst_corrected[:n_samples, :, :],
                        'IFS_raw': fcst_array[:n_samples, :, :]}

    # Calculate FSS
    hourly_thresholds = np.quantile(truth_array, quantile_locs)
    fss_results_path = os.path.join(model_eval_folder, f'fss_{model_number}_{area}.pkl')

    if os.path.exists(fss_results_path):
        with open(fss_results_path, 'rb') as ifh:
            fss_results = pickle.load(ifh)
    else:
        
        fss_results = get_fss_scores(truth_array, fss_data_dict, hourly_thresholds, scales, n_samples)
        with open(fss_results_path, 'wb+') as ofh:
            pickle.dump(fss_results, ofh)

    # Plot FSS
    plt.rcParams.update({'font.size': 11})
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]
    n=0

    num_cols = 3
    num_rows = 2
    fig = plt.figure(constrained_layout=True, figsize=(4*num_cols, 4*num_rows))
    gs = gridspec.GridSpec(num_rows, 2*num_cols, figure=fig, 
                    wspace=0.005)  
        

    fss_format_lookup = {'fcst': {'label': 'IFS-qm', 'color': 'r'}, 
                        'cgan':  {'label': 'cGAN-qm', 'color': 'b'},
                        'cgan_raw': {'label': 'cGAN', 'color': 'b', 'linestyle': '--'},
                        'fcst_raw': {'label': 'IFS', 'color': 'r', 'linestyle': '--'}}


    for n, thr in enumerate(fss_bstrap_results['cgan'].keys()):
        
        col = n%3
        row = int(n/3)
        
        if row == 1:
            ax = fig.add_subplot(gs[row, 2*col+1:2*col+3])
        else:
            ax = fig.add_subplot(gs[row, 2*col:2*col+2])
        q_val_text = f'{np.round(float(thr)*100, 12)}th'.replace('.0th', 'th')

        
        for m, (name) in enumerate(fss_bstrap_results.keys()):
            label = fss_format_lookup[name]['label']
                

            bstrap_data = fss_bstrap_results[name][thr]
            window_sizes = list(bstrap_data.keys())
            window_sizes_km = 11* np.array(window_sizes)
            mean_fss_values = [item['mean'] for item in bstrap_data.values()]
            std_fss_values = np.array([item['std'] for item in bstrap_data.values()])

            ax.plot(window_sizes_km, mean_fss_values, 'o', 
                    color=fss_format_lookup[name]['color'], 
                    label=fss_format_lookup[name]['label'], 
                    markersize=5,
                    linestyle=fss_format_lookup[name].get('linestyle', '-'))
            
            if name in ['cgan', 'fcst']:
                ax.fill_between(window_sizes_km, y1=mean_fss_values-2*std_fss_values, y2=mean_fss_values+2*std_fss_values,
                                    color=fss_format_lookup[name].get('color', 'black'), alpha=0.2)

        ax.set_title(f'({ascii_lowercase[n]}) {q_val_text}')

        ax.set_ylim(0,1)
        ax.set_xlabel('Neighbourhood width (km)')
        ax.set_ylabel('FSS')
        ax.set_xlim(0, 1 + 11*max(fss_bstrap_results[name][thr].keys()))
        if thr in [0.999, 0.99999, 0.9999]:
            ax.legend(loc='upper right', frameon=False, ncol=2)
        else:
            ax.legend(loc='lower right', frameon=False, ncol=2)

    plot_filename = os.path.join(args.output_dir, f'fss_{model_number}_{area}.pdf')

    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')

#################################################################################

if metric_dict['diurnal']:
    print('*********** Plotting Diurnal cycle **********************')

    
    diurnal_data_dict = {'Obs (IMERG)': truth_array,
                         'cGAN': cgan_corrected[:,:,:,0],
                         'IFS': fcst_corrected
                         }
    ensemble_diurnal_data_dict = {'Obs (IMERG)': truth_array,
                         'cGAN': cgan_corrected,
                         'IFS': fcst_corrected
                         }
    format_lkp = {'cGAN': {'color': 'b'}, 'IFS': {'color': 'r'}, 'Obs (IMERG)': {}}

    metric_types = {'quantile_999': lambda x,y: np.quantile(x, 0.999),
                    'quantile_9999': lambda x,y: np.quantile(x, 0.9999),
                    'quantile_99999': lambda x,y: np.quantile(x, 0.99999),
                'median': lambda x,y: np.quantile(x, 0.5),
                'mean': lambda x,y: np.mean(x)}
    
  
    plot_data = {}
    
    for metric_name in ['mean', 'quantile_999', 'quantile_9999', 'quantile_99999']:
        
        metric_fn = metric_types[metric_name]
        plot_data[metric_name] = {}
        
        for name, arr in ensemble_diurnal_data_dict.items():
            if name == 'cGAN':
                cgan_metrics_by_hour = []
                for n in tqdm(range(arr.shape[-1])):
                    metric_by_hour, hour_bin_edges = get_metric_by_hour(metric_fn, 
                                                                        obs_array=arr[:,:,:,n], 
                                                                        fcst_array=arr[:,:,:,n], 
                                                                        hours=hours, 
                                                                        bin_width=3)
                    cgan_metrics_by_hour.append(metric_by_hour)
                cgan_metric_mean = np.mean(np.stack([list(item.values()) for item in cgan_metrics_by_hour]), axis=0)
                cgan_metric_std = np.std(np.stack([list(item.values()) for item in cgan_metrics_by_hour]), axis=0)
                cgan_metric_max = np.max(np.stack([list(item.values()) for item in cgan_metrics_by_hour]), axis=0)
                cgan_metric_min = np.min(np.stack([list(item.values()) for item in cgan_metrics_by_hour]), axis=0)

                plot_data[metric_name][name] = {'cgan_metric_mean': cgan_metric_mean,
                                    'cgan_metric_std': cgan_metric_std,
                                    'cgan_metric_max': cgan_metric_max,
                                    'cgan_metric_min': cgan_metric_min}
            else:
                metric_by_hour, hour_bin_edges = get_metric_by_hour(metric_fn, obs_array=arr, fcst_array=arr, hours=hours, bin_width=3)
                plot_data[metric_name][name] = metric_by_hour
                
    with open(os.path.join(args.output_dir, f'diurnal_cycle_{nickname}_{model_number}.pkl'), 'wb+') as ofh:
        pickle.dump(plot_data, ofh)
        
    format_lkp = {'cGAN': {'color': 'b', 'label': 'cGAN-qm'}, 'IFS': {'color': 'r', 'label': 'IFS-qm'}, 'Obs (IMERG)': {'label': 'IMERG'}}
    plt.rcParams.update({'font.size': 15})

    x_vals = list(plot_data['mean']['IFS'].keys())
    bin_width = 24 / len(x_vals)
    hour_bin_edges = np.arange(0, 24, bin_width)

    for metric_name, diurnal_dict in plot_data.items():
        fig, ax = plt.subplots(1,1, figsize=(7,6))

        for name, arr in diurnal_dict.items():
            label=format_lkp[name]['label']
            if name == 'cGAN':
            
                ax.plot(x_vals, arr['cgan_metric_mean'], '-o', label=label, color=format_lkp[name].get('color', 'black'))
                ax.fill_between(x_vals, arr['cgan_metric_min'], arr['cgan_metric_max'], alpha=0.4)
            else:
                ax.plot(x_vals, arr.values(), '-o', label=label, color=format_lkp[name].get('color', 'black'))
            
        max_val = np.max(diurnal_dict['IFS'])
        ax.set_xticks(np.array(x_vals) - .5)
        ax.set_xticklabels([int(hr) for hr in hour_bin_edges])
        ax.set_xlabel('Hour')
        ax.set_ylabel('Average mm/hr')
        ax.set_ylim([0, None])
        ax.legend(loc='lower right')

        plt.savefig(os.path.join(args.output_dir,f'diurnal_cycle_{metric_name}_{nickname}_{model_number}_{area}.pdf'), bbox_inches='tight')

        
    
    # # Seasonal diurnal cycle
    # seasons_dict = {'MAM': [3,4,5], 'OND': [10,11,12]}
    # hourly_season_sums, hourly_season_counts = {}, {}

    # hourly_season_sums = {}
    # for k, d in diurnal_data_dict.items():
    #     hourly_season_sums[k] = {}
    #     for n, (season, month_range) in enumerate(seasons_dict.items()):

    #         hourly_season_sums[k][season], hourly_season_counts[season] = get_diurnal_cycle(d, 
    #                                                                                         dates, hours, 
    #                                                                                         longitude_range=longitude_range,
    #                                                                                         latitude_range=latitude_range)

    # fig, ax = plt.subplots(len(seasons_dict),1, figsize=(12,12))
    # fig.tight_layout(pad=3)

    # for n, season in enumerate(seasons_dict):
    #     for name, data in hourly_season_sums.items():
            
    #         mean_hourly_data = [np.mean(data[season][n] / hourly_season_counts[season][n]) for n in range(23)]
    #         ax[n].plot(mean_hourly_data, '-o',label=name)
        
    #     ax[n].legend()
    #     ax[n].set_xlabel('Hour')
    #     ax[n].set_ylabel('Average mm/hr')
    #     ax[n].set_title(season)
    # plt.savefig(f'plots/diurnal_cycle_seasonal_{model_type}_{model_number}.pdf')
    
    # with open(f'plots/diurnal_cycle_seasonal_{model_type}_{model_number}.pkl', 'wb+') as ofh:
    #     pickle.dump(diurnal_data_dict, ofh)
                
    # Diurnal maximum map
    import xarray as xr
    from datetime import datetime
    from dsrnngan.utils.utils import get_local_hour
    from scipy.ndimage import uniform_filter, uniform_filter1d
    import xarray as xr


    hour_bin_edges = np.arange(0, 24, 1)
    filter_size = 31 # roughly the size of lake victoria region
    digitized_hours = np.digitize(hours, bins=hour_bin_edges)

    n_samples = truth_array.shape[0]
    
    raw_diurnal_data_dict = {'IMERG': truth_array,
                        'cGAN-qm':cgan_corrected[:,:,:,0],
                        'IFS-qm': fcst_corrected}

    peak_dict = {}

    for name in raw_diurnal_data_dict:
        
        da = xr.DataArray(
            data=raw_diurnal_data_dict[name],
            dims=["hour_range", "lat", "lon"],
            coords=dict(
                lon=(longitude_range),
                lat=(latitude_range ),
                hour_range=digitized_hours,
            ),
            attrs=dict(
                description="Precipitation.",
                units="mm/hr",
            ),
        )


        smoothed_vals = uniform_filter1d(da.groupby('hour_range').mean().values, 3, axis=0, mode='wrap')
        peak_dict[name] = np.argmax(smoothed_vals, axis=0)

    from dsrnngan.evaluation import plots
    from matplotlib.pyplot import cm
    from matplotlib import colors
    plt.rcParams.update({'font.size': 9})

    n_cols = len(peak_dict)
    n_rows = 1
    fig = plt.figure(constrained_layout=True, figsize=(2.5*n_cols, 3*n_rows))
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, 
                        width_ratios=[1]*(n_cols ),
                        height_ratios=[1]*(n_rows) + [0.05],
                    wspace=0.005) 

    ax = fig.add_subplot(gs[0, 0], projection = ccrs.PlateCarree())

    filter_size=1


    n=1
    for name, peak_data in peak_dict.items():
        if name != 'IMERG':
            ax = fig.add_subplot(gs[0, n], projection = ccrs.PlateCarree())
            smoothed_peak_data = uniform_filter(peak_data.copy(), size=filter_size, mode='reflect')
            
            im1 = ax.imshow(np.flip(smoothed_peak_data, axis=0), extent = [ min(longitude_range), max(longitude_range), min(latitude_range), max(latitude_range)], 
                    transform=ccrs.PlateCarree(), cmap='twilight_shifted', vmin=0, vmax=23)

            
            n+=1
        else:
            smoothed_peak_data = uniform_filter(peak_data.copy(), size=filter_size, mode='reflect')

            name = 'IMERG'
            im_imerg = ax.imshow(np.flip(smoothed_peak_data, axis=0), extent = [ min(longitude_range), max(longitude_range), min(latitude_range), max(latitude_range)], 
                    transform=ccrs.PlateCarree(), cmap='twilight_shifted', vmin=0, vmax=23)
        print(name, peak_data.max())
        ax.add_feature(plots.border_feature)
        ax.add_feature(plots.disputed_border_feature)
        ax.add_feature(plots.lake_feature, alpha=0.4)
        ax.coastlines(resolution='10m', color='black', linewidth=0.4)
        ax.set_title(name)
            
    cbar_ax = fig.add_subplot(gs[-1, 1])
    # cb = fig.colorbar(cm.ScalarMappable(norm=None, cmap=colors.Colormap('twilight_shifted')), cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)))
    cb = fig.colorbar(im_imerg, cax=cbar_ax, orientation='horizontal', shrink = 0.2, aspect=10, ticks=range(len(hour_bin_edges)),
                        )
    cb.ax.set_xticks(range(0,24,3))
    cb.ax.set_xticklabels(range(0,24,3))
    cb.ax.set_xlabel("Peak rainfall hour (EAT)", loc='center')
    plt.savefig(os.path.join(args.output_dir,f'diurnal_cycle_map_{nickname}_{model_number}_{area}.pdf'), format='pdf')
    
#################################################################################

if metric_dict.get('confusion_matrix'):
    print('*********** Calculating confusion matrices **********************')

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    
    # quantile_locations = [0.1, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    # hourly_thresholds = np.quantile(truth_array, quantile_locations)
    hourly_thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    results = {'hourly_thresholds': hourly_thresholds,
               'conf_mat': []}
    
    for threshold in hourly_thresholds:
        y_true = (truth_array > threshold).astype(np.int0).flatten()

        y_dict = {
                'cgan_qmap' : (cgan_corrected[:,:,:,0]> threshold).astype(np.int0).flatten(),
                'ifs_qmap': (fcst_corrected > threshold).astype(np.int0).flatten(),
                'cgan_ens': (np.mean(cgan_corrected, axis=-1)> threshold).astype(np.int0).flatten()}

        tmp_results_dict = {'threshold': threshold}
        for k, v in tqdm(y_dict.items()):
            tmp_results_dict[k] = confusion_matrix(y_true, v)
        
        results['conf_mat'].append(tmp_results_dict)
        with open(os.path.join(args.output_dir,f'confusion_matrices_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
            pickle.dump(results, ofh)
            
if metric_dict.get('csi'):
    print('*********** Calculating critical success index **********************')

    from dsrnngan.evaluation import scoring
    hourly_thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    csi_dict = {
                         'GAN': samples_gen_array[:,:,:,0],
                         'IFS': fcst_array,
                         'GAN + qmap': cgan_corrected[:,:,:,0],
                         'IFS + qmap': fcst_corrected
                         }
    
    csi_results =  scoring.get_skill_score_results(
            skill_score_function=scoring.critical_success_index,
                     data_dict=csi_dict, obs_array=truth_array,
                     hours=hours,
                     hourly_thresholds=hourly_thresholds
                    )
        
    with open(os.path.join(args.output_dir,f'csi_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(csi_results, ofh)
        

    ets_results =  scoring.get_skill_score_results(
            skill_score_function=scoring.equitable_threat_score,
                     data_dict=csi_dict, obs_array=truth_array,
                    hours=hours,
                    hourly_thresholds=hourly_thresholds
                    )
    
    with open(os.path.join(args.output_dir, f'ets_{nickname}_{model_number}_{area}.pkl'), 'wb+') as ofh:
        pickle.dump(ets_results, ofh)
    

    
        