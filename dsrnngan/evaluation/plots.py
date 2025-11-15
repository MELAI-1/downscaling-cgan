import os
import pickle
from tqdm import tqdm

import copy
import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import seaborn as sns
from typing import Iterable, List
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
from metpy import plots as metpy_plots
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, auto_scaler, AdaptiveScaler
from shapely.geometry import Polygon, LineString

# See https://matplotlib.org/stable/gallery/color/named_colors.html for edge colour options
lake_feature = cfeature.NaturalEarthFeature(
    'physical', 'lakes',
    cfeature.auto_scaler, edgecolor='black', facecolor='never')

# Have to subclass the borders feature to remove controversial borders
class EABorderFeature(NaturalEarthFeature):
            
    def __init__(self, category: str, name: str, scale: AdaptiveScaler, border_to_remove: list, **kwargs):
        self.removed_borders_poly = Polygon(border_to_remove)
        
        super().__init__(category, name, scale, **kwargs)
    
    def geometries(self):
        geoms =  list(super().geometries())

        borders = [item for item in geoms if not self.removed_borders_poly.contains(item)]

        return iter(borders)
    
class DisputedBorders(NaturalEarthFeature):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def geometries(self):
        borders = [LineString([[34.37, 4.63], [35.94, 4.59]]), LineString([[34.37, 4.63], [35.30, 5.47]])]
        return iter(borders)
    
border_feature = EABorderFeature(
                                    'cultural', 
                                    'admin_0_boundary_lines_land',
                                     auto_scaler, 
                                     border_to_remove=[[47.5,7.85], [47.86, 7.52], [52.54, 10.26], [47.08, 12.7]], 
                                     edgecolor='black', 
                                     facecolor='never')

disputed_border_feature = DisputedBorders('cultural', 
                                    'admin_0_boundary_lines_land',
                                     auto_scaler, 
                                     edgecolor='black', 
                                     facecolor='never',
                                     linestyle='--')

from dsrnngan.utils import read_config, utils
from dsrnngan.data import data
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.evaluation.rapsd import plot_spectrum1d, rapsd
from dsrnngan.evaluation.thresholded_ranks import findthresh
from dsrnngan.evaluation import scoring

path = os.path.dirname(os.path.abspath(__file__))

palette="YlGnBu"
default_linewidth = 0.4
default_cmap = ListedColormap(sns.color_palette(palette, 256))
default_cmap.set_under('white')
default_latitude_range, default_longitude_range = read_config.get_lat_lon_range_from_config()
default_extent = [min(default_longitude_range), max(default_longitude_range), 
                  min(default_latitude_range), max(default_latitude_range)]
alpha = 0.8
spacing = 10

cmap = ListedColormap(sns.color_palette(palette, 256))
cmap.set_under('white')

step_size = 0.001
range_dict = {0: {'start': 0.1, 'stop': 1, 'interval': 0.1, 'marker': '+', 'marker_size': 32},
              1: {'start': 1, 'stop': 10, 'interval': 1, 'marker': '+', 'marker_size': 256},
              2: {'start': 10, 'stop': 80, 'interval':1, 'marker': '+', 'marker_size': 256},
              3: {'start': 80, 'stop': 99.1, 'interval': 1, 'marker': '+', 'marker_size': 256},
              4: {'start': 99.1, 'stop': 99.91, 'interval': 0.1, 'marker': '+', 'marker_size': 128},
              5: {'start': 99.9, 'stop': 99.99, 'interval': 0.01, 'marker': '+', 'marker_size': 32 },
              6: {'start': 99.99, 'stop': 99.999, 'interval': 0.001, 'marker': '+', 'marker_size': 10},
              7: {'start': 99.999, 'stop': 99.9999, 'interval': 0.0001, 'marker': '+', 'marker_size': 10},
              8: {'start': 99.9999, 'stop': 99.99999, 'interval': 0.00001, 'marker': '+', 'marker_size': 10}}

percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
percentiles=np.concatenate(percentiles_list)
quantile_locs = [item / 100.0 for item in percentiles]


def get_geoaxes(*args, **kwargs):
    fig, ax = plt.subplots(*args, 
                        subplot_kw={'projection' : ccrs.PlateCarree()},
                        **kwargs)
    
    return fig, ax

def plot_contourf( data, title, ax=None, value_range=None, lon_range=default_longitude_range, lat_range=default_latitude_range,
                  cmap='Reds', extend: str='both', add_borders=True):
    
    if ax is None:
        fig, ax = get_geoaxes(1,1)
    
    if value_range is not None:
        im = ax.contourf(lon_range, lat_range, data, transform=ccrs.PlateCarree(),
                            cmap=cmap, 
                            levels=value_range, norm=colors.Normalize(min(value_range), max(value_range)),
                            extend=extend)
    else:

        im = ax.contourf(lon_range, lat_range, data, transform=ccrs.PlateCarree(),
                    cmap=cmap, 
                    extend=extend)

    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
    
    if add_borders:
        ax.add_feature(border_feature)
        ax.add_feature(disputed_border_feature)
    ax.add_feature(lake_feature, alpha=0.4)
    ax.set_title(title)
    
    return im


def plot_fss_scores(fss_results, output_folder, output_suffix):
    fig, axs = plt.subplots(len(fss_results['thresholds']), 1, figsize = (10, 20))
    fig.tight_layout(pad=4.0)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]

    for n, thr in enumerate(fss_results['thresholds']):
        
        for m, (name, scores) in enumerate(fss_results['scores'].items()):
            axs[n].plot(fss_results['window_sizes'], scores[n], label=name, color='k', linestyle=linestyles[m])
            

        axs[n].set_title(f'Threshold = {thr:0.1f} mm/hr')

    for ax in axs:    
        ax.hlines(0.5, 0, max(fss_results['window_sizes']), linestyles='dashed', colors=['r'])
        ax.set_ylim(0,1)
        ax.set_xlabel('Neighbourhood size')
        ax.set_ylabel('FSS')
        ax.set_xlim(0, max(fss_results['window_sizes']))
        ax.legend()
        
    plt.savefig(os.path.join(output_folder, f'fractional_skill_score_{output_suffix}.pdf'), format='pdf')
    

def get_quantile_data(quantile_data_dict: dict,
                   save_path: str=None, 
                   range_dict: dict=range_dict):
    """
    Produce qauntile-quantile plot

    Args:
        quantile_data_dict (dict): Dict containing entries for data sets to plot. One must have the key=obs_key. Structure:
                    {
                    data_name_1:  np.ndarray,
                    data_name_2:  np.ndarray
                    }
        save_path (str, optional): Path to save plots in. Defaults to None.
        range_dict (dict, optional): Dict containing ranges of quantiles. Defaults to range_dict.

    Returns:
        fig, ax: Figure and axis with plotted data.
    """

    quantile_results = {}
    quantile_values = {}
    intervals = {}
    
    for data_name, d in quantile_data_dict.items():
        quantile_results[data_name] = {}
        
        for k, v in tqdm(range_dict.items()):
        
            boundaries = np.arange(v['start'], v['stop'], v['interval']) / 100
            
            if len(boundaries) > 0:
                quantile_results[data_name][k] = np.quantile(d, boundaries)
                quantile_values[k] = boundaries

    # Save raw data
    if save_path:
        format_str = '.' + save_path.split('.')[-1]
        data_save_path = save_path.replace(format_str, '.pkl')
        
        with open(data_save_path, 'wb+') as ofh:
            pickle.dump({'quantile_results': quantile_results,
                         'quantile_values': quantile_values,
                         'interval_list': intervals}, ofh)
            
    return quantile_results, quantile_values, intervals
    
            
            
def plot_quantiles(quantile_results: dict, 
                   quantile_data_dict: dict,
                   format_lookup: dict,
                   save_path: str=None, fig: plt.figure=None, 
                   ax: plt.Axes=None, 
                   obs_key: str='Obs (IMERG)',
                   range_dict: dict=range_dict,
                   max_quantile: float=1.0,
                   min_data_points_per_quantile: int=50,
                   format_str='.pdf'):
    """
    Produce qauntile-quantile plot

    Args:
        quantile_results (dict): Output of get_quantile_data function
        format_lookup (dict): Doct containing formatting details for different datasets
            e.g.   {data_name_1: {'color': str, 'marker': str, 'alpha': float} }
        save_path (str, optional): Path to save plots in. Defaults to None.
        fig (plt.figure, optional): Existing figure to plot in. Defaults to None.
        ax (plt.Axes, optional): Axis to plot on. Defaults to None.
        obs_key (str, optional): Key that corresponds to observations. Defaults to 'Obs (IMERG)'.
        range_dict (dict, optional): Dict containing ranges of quantiles. Defaults to range_dict.

    Returns:
        fig, ax: Figure and axis with plotted data.
    """
    
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    marker_handles = None

    # Quantiles for annotating plot
    quantile_annotation_dict = {str(np.round(q, 11)): np.quantile(quantile_data_dict[obs_key], q) for q in [1 - 10**(-n) for n in range(3, 6)] if q <=max_quantile}
        
    for k, v in tqdm(range_dict.items()):
        
        # size= v.get('marker_size', 32)
        size = 20
        cmap = plt.colormaps["plasma"]
        
        if k in quantile_results[obs_key]:
            max_truth_val = max(quantile_results[obs_key][k])
            
            marker_hndl_list = []
            for data_name, res in quantile_results.items():
                if data_name != obs_key:
                    s = ax.scatter(quantile_results[obs_key][k], 
                                   res[k], 
                                   c=format_lookup[data_name]['color'],
                                   marker=format_lookup[data_name]['marker'], 
                                   label=data_name, 
                                   s=size, 
                                cmap=cmap, alpha=format_lookup[data_name].get('alpha', 1))
                    marker_hndl_list.append(s)
            
            if not marker_handles:
                marker_handles = marker_hndl_list
        
    
    ax.plot(np.linspace(0, max_truth_val, 100), np.linspace(0, max_truth_val, 100), 'k--')
    ax.set_xlabel('Observations (mm/hr)')
    ax.set_ylabel('Model (mm/hr)')
    ax.legend(handles=marker_handles, loc='lower right')
    
    # find largest value
    max_val = 0
    for v in quantile_results.values():
        for sub_v in v.values():
            if max(sub_v) > max_val:
                max_val = max(sub_v)
    
    for k, v in quantile_annotation_dict.items():
        ax.vlines(v, 0, max_val, linestyles='--')
        ax.text(1.01*v, 0.95* max_val, f'{np.round(float(k)*100, 12)}th')
    
    if save_path:
        plt.savefig(save_path, format=format_str.replace('.', ''), bbox_inches='tight')
        
    return fig, ax


def plot_precipitation(ax: plt.Axes, 
                       data: np.ndarray, 
                       title:str, 
                       longitude_range: Iterable[float]=default_longitude_range, 
                       latitude_range: Iterable[float]=default_latitude_range,
                       precip_cmap=None,
                       precip_norm=None,
                       levels: List[int]=None):
    if levels is None:
        levels = [0, 0.1, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150]
    
    if precip_cmap is None:
        precip_cmap = ListedColormap(metpy_plots.ctables.colortables["precipitation"][:len(levels)-1], 'precipitation')
    if precip_norm is None:
        precip_norm = BoundaryNorm(levels, precip_cmap.N)

    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
            
    im = ax.imshow(data,
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
    ax.add_feature(lake_feature, alpha=0.4)
    ax.set_title(title)
    
    return im

def plot_csi(data_dict: dict, obs_array: np.ndarray,
             hours: Iterable, format_lookup: dict,
             hourly_thresholds: list= [0.1, 5, 20],
             obs_key: str='Obs (IMERG)',
             save_path: str=None):
    
    csi_results = []
        
    for threshold in hourly_thresholds:
        
        tmp_results_dict = {'threshold': threshold}
        for k, v in tqdm(data_dict.items()):
            tmp_results_dict[k] = scoring.critical_success_index(obs_array, v, threshold=threshold)
            
            metric_fn = lambda x,y: scoring.critical_success_index(x, y, threshold=threshold)
            
            tmp_results_dict[k + '_hourly'] = scoring.get_metric_by_hour(metric_fn=metric_fn, obs_array=obs_array, fcst_array=v, 
                                                    hours=hours,
                                                    bin_width=1)
        
        csi_results.append(tmp_results_dict)
    
    if save_path:
        format_str = '.' + save_path.split('.')[-1]
        data_save_path = save_path.replace(format_str, '.pkl')
        with open(data_save_path, 'wb+') as ofh:
            pickle.dump(csi_results, ofh)

    fig, axs = plt.subplots(len(data_dict),1, figsize=(8, 8))
    for n in range(len(data_dict)):
        for pk in data_dict:
            
            axs[n].plot(csi_results[n][pk + '_hourly'][0].values(), label=pk, **format_lookup[pk])
            axs[n].set_title(f"Threshold = {csi_results[n]['threshold']}")
        axs[n].legend()
        axs[n].set_ylim([0,None])

    fig.tight_layout(pad=1)
        
    if save_path:
        plt.savefig(save_path, format=format_str.replace('.', ''))

def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.Normalize(*value_range),
               origin='lower',
               extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def plot_img_log(img, value_range=(0.01, 5), extent=None):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.LogNorm(*value_range),
               origin='lower',
               extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def plot_img_log_coastlines(img, value_range_precip=(0.01, 5), cmap='viridis', extent=None, alpha=0.8):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.LogNorm(*value_range_precip),
               cmap=cmap,
               origin='lower',
               extent=extent,
               transform=ccrs.PlateCarree(),
               alpha=alpha)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def truncate_colourmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_sequences(gen,
                   mode,
                   batch_gen,
                   checkpoint,
                   noise_channels,
                   latent_variables,
                   num_samples=8,
                   num_instances=4,
                   out_fn=None):

    for cond, const, seq_real in batch_gen.as_numpy_iterator():
        batch_size = cond.shape[0]

    seq_gen = []
    if mode == 'GAN':
        for i in range(num_instances):
            noise_shape = cond[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.predict([cond, const, noise_gen()]))
    elif mode == 'det':
        for i in range(num_instances):
            seq_gen.append(gen.predict([cond, const]))
    elif mode == 'VAEGAN':
        # call encoder
        (mean, logvar) = gen.encoder([cond, const])
        # run decoder n times
        for i in range(num_instances):
            noise_shape = cond[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.decoder.predict([mean, logvar, noise_gen(), const]))

    seq_real = data.denormalise(seq_real)
    cond = data.denormalise(cond)
    seq_gen = [data.denormalise(seq) for seq in seq_gen]

    num_rows = num_samples
    num_cols = 2+num_instances

    figsize = (num_cols*1.5, num_rows*1.5)
    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows, num_cols,
                           wspace=0.05, hspace=0.05)

    value_range = (0, 5)  # batch_gen.decoder.value_range

    for s in range(num_samples):
        i = s
        plt.subplot(gs[i, 0])
        plot_img(seq_real[s, :, :, 0], value_range=value_range)
        plt.subplot(gs[i, 1])
        plot_img(cond[s, :, :, 0], value_range=value_range)
        for k in range(num_instances):
            j = 2+k
            plt.subplot(gs[i, j])
            plot_img(seq_gen[k][s, :, :, 0], value_range=value_range)

    plt.suptitle('Checkpoint ' + str(checkpoint))

    if out_fn is not None:
        # plt.savefig(out_fn, bbox_inches='tight')
        # plt.savefig(out_fn + f'-ckpt{checkpoint}.pdf', bbox_inches='tight')
        plt.savefig(out_fn+f"_{checkpoint}.pdf", bbox_inches='tight')
        plt.close()


def plot_rank_histogram(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0, 1, N_ranks)
    db = (bc[1] - bc[0])
    bins = bc - db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h, _) = np.histogram(ranks, bins=bins)
    h = h / h.sum()

    ax.plot(bc, h, **plot_params)


def plot_rank_cdf(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0, 1, N_ranks)
    db = (bc[1] - bc[0])
    bins = bc - db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h, _) = np.histogram(ranks, bins=bins)
    h = h.cumsum()
    h = h / h[-1]

    ax.plot(bc, h, **plot_params)


def plot_rank_histogram_all(rank_files,
                            labels,
                            log_path,
                            N_ranks=101,
                            threshold=False,
                            freq=0.0001,
                            lead_time=None,
                            model=None,
                            ablation=False):
    (fig, axes) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
    if lead_time is not None:
        if threshold:
            fig.suptitle('Rank histograms {} - top {:.2%}'.format(model, freq))
        else:
            fig.suptitle('Rank histograms {} - all'.format(model))
    else:
        if threshold:
            fig.suptitle('Rank histograms - top {:.2%}'.format(freq))
        else:
            fig.suptitle('Rank histograms - all')
    plt.subplots_adjust(hspace=0.15)
    plt.rcParams['font.size'] = '12'
    ls = "solid"

    for (fn_valid, label) in zip(rank_files, labels):
        with np.load(fn_valid) as f:
            ranks = f['ranks']
            if threshold:
                lowres = f['lowres']
                assert ranks.shape == lowres.shape
                thresh = findthresh(lowres, freq).root  # find IFS tp threshold
                ranks = ranks[np.where(lowres > thresh)]  # restrict to these events

        plot_rank_histogram(axes[0], ranks, N_ranks=N_ranks, label=label, linestyle=ls, linewidth=0.75, zorder=2)
        plot_rank_cdf(axes[1], ranks, N_ranks=N_ranks, label=label, linestyle=ls, linewidth=0.75, zorder=2)

    bc = np.linspace(0, 1, N_ranks)
    axes[0].plot(bc, [1./N_ranks]*len(bc), linestyle=':', label="Uniform", c='dimgrey', zorder=0)
    axes[0].set_ylabel("Norm. occurrence")
    ylim = axes[0].get_ylim()
    if threshold and lead_time:
        axes[0].set_ylim((0, 0.1))
    elif lead_time:
        axes[0].set_ylim((0, 0.05))
    else:
        axes[0].set_ylim((0, ylim[1]))
    axes[0].set_xlim((0, 1))
    axes[0].text(0.01, 0.97, "(a)",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=axes[0].transAxes)

    axes[1].plot(bc, bc, linestyle=':', label="Ideal", c='dimgrey', zorder=0)
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("Normalized rank")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlim((0, 1))
    axes[1].text(0.01, 0.97, "(b)",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=axes[1].transAxes)
    axes[1].legend(bbox_to_anchor=(1.05, 1.05))

    if lead_time is not None:
        if threshold:
            plt.savefig("{}/rank-distribution-{}-{}-{}.pdf".format(log_path, 'lead_time', model, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution-{}-{}.pdf".format(log_path, 'lead_time', model), bbox_inches='tight')
    elif ablation:
        if threshold:
            plt.savefig("{}/rank-distribution-ablation-{}.pdf".format(log_path, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution-ablation.pdf".format(log_path), bbox_inches='tight')
    else:
        if threshold:
            plt.savefig("{}/rank-distribution-{}.pdf".format(log_path, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution.pdf".format(log_path), bbox_inches='tight')
    plt.close()


def plot_histograms(log_folder, val_range, ranks, N_ranks):
    rank_metrics_files = [os.path.join(log_folder, f"ranks-{'-'.join(val_range)}_{rk}.npz") for rk in ranks]
    labels = [f"{rk}" for rk in ranks]
    plot_rank_histogram_all(rank_metrics_files, labels, log_folder, N_ranks=N_ranks)


def gridplot(models, model_labels=None,
             vmin=0, vmax=1):
    nx = models[0].shape[0]
    ny = len(models)
    fig = plt.figure(dpi=200, figsize=(nx, ny))
    gs1 = gridspec.GridSpec(ny, nx)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    for i in range(nx):
        for j in range(ny):
            # print(i,j)
            ax = plt.subplot(gs1[i+j*nx])  # plt.subplot(ny,nx,i+1+j*nx)
            ax.pcolormesh(models[j][i, :, :], vmin=vmin, vmax=vmax)
            # ax.axis('off')
            ax.set(xticks=[], yticks=[])
            if i == 0 and (model_labels is not None):
                ax.set_ylabel(model_labels[j])
            ax.axis('equal')
    fig.text(0.5, 0.9, 'Dates', ha='center')
    fig.text(0.04, 0.5, 'Models', va='center', rotation='vertical')
    return


def plot_roc_curves(roc_files,
                    labels,
                    log_path,
                    precip_values,
                    pooling_methods,
                    lw=2):

    roc_data = {}
    for (fn, label) in zip(roc_files, labels):
        with open(fn, 'rb') as handle:
            roc_data[label] = pickle.load(handle)

    for method in pooling_methods:
        for i in range(len(precip_values)):
            plt.plot(roc_data['IFS_fpr'][method][i],
                     roc_data['IFS_tpr'][method][i],
                     linestyle="",
                     marker="x",
                     label="IFS")
            plt.plot(roc_data['GAN_fpr'][method][i],
                     roc_data['GAN_tpr'][method][i],
                     lw=lw,
                     label="GAN (area = %0.2f)" % roc_data['GAN_auc'][method][i])
            plt.plot(roc_data['VAEGAN_fpr'][method][i],
                     roc_data['VAEGAN_tpr'][method][i],
                     lw=lw,
                     label="VAEGAN (area = %0.2f)" % roc_data['VAEGAN_auc'][method][i])

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
            plt.plot([], [], ' ',
                     label="event frequency %0.3f" % roc_data['GAN_base'][method][i])  # plot base rate
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate,  FP/(FP+TN)')
            plt.ylabel('True Positive Rate, TP/(TP+FN)')
            plt.title(f'ROC curve for >{precip_values[i]}mm, {method}')
            plt.legend(loc="lower right")
            plt.savefig("{}/ROC-{}-{}.pdf".format(log_path, precip_values[i], method), bbox_inches='tight')
            plt.close()


def plot_prc_curves(prc_files,
                    labels,
                    log_path,
                    precip_values,
                    pooling_methods,
                    lw=2):

    prc_data = {}
    for (fn, label) in zip(prc_files, labels):
        with open(fn, 'rb') as handle:
            prc_data[label] = pickle.load(handle)

    for method in pooling_methods:
        for i in range(len(precip_values)):
            plt.plot(prc_data['IFS_rec'][method][i],
                     prc_data['IFS_pre'][method][i],
                     linestyle="",
                     marker="x",
                     label="IFS")
            plt.plot(prc_data['GAN_rec'][method][i],
                     prc_data['GAN_pre'][method][i],
                     lw=lw,
                     label="GAN (area = %0.2f)" % prc_data['GAN_auc'][method][i])
            plt.plot(prc_data['VAEGAN_rec'][method][i],
                     prc_data['VAEGAN_pre'][method][i],
                     lw=lw,
                     label="VAEGAN (area = %0.2f)" % prc_data['VAEGAN_auc'][method][i])
            plt.plot([0, 1],
                     [prc_data['GAN_base'][method][i], prc_data['GAN_base'][method][i]],
                     '--',
                     lw=0.5,
                     color='gray',
                     label="event frequency %0.3f" % prc_data['GAN_base'][method][i])  # no skill
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall TP/(TP+FN)')
            plt.ylabel('Precision TP/(TP+FP)')
            plt.title(f'Precision-recall curve for >{precip_values[i]}mm, {method}')
            plt.legend()
            plt.savefig("{}/PR-{}-{}.pdf".format(log_path, precip_values[i], method), bbox_inches='tight')
            plt.close()


def plot_fss(fss_files,
             labels,
             log_path,
             nimg,
             precip_values,
             spatial_scales,
             full_image_npixels,
             lw=2):

    for i in range(len(precip_values)):
        baserate_first = None
        plt.figure(figsize=(7, 5), dpi=200)
        plt.gcf().set_facecolor("white")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # load data
        j = 0
        for (fn, label, color) in zip(fss_files, labels, colors):
            f1 = fn + '-1.pickle'
            f2 = fn + '-2.pickle'
            with open(f1, 'rb') as f:
                m1data = pickle.load(f)[precip_values[i]]
            with open(f2, 'rb') as f:
                m2data = pickle.load(f)[precip_values[i]]

            assert spatial_scales == list(m1data.keys()), "spatial scales do not match"

            y1 = [m1data[spasc]["score"] for spasc in spatial_scales]
            y2 = [m2data[spasc]["score"] for spasc in spatial_scales]
            plt.semilogx(spatial_scales, y1, '-', color=color, lw=lw,
                         label=f"{labels[j]}")
            plt.semilogx(spatial_scales, y2, ':', color=color, lw=lw)

            # obtain base frequency for no-skill and target-skill lines
            baserate = m1data[1]['fssobj']['sum_obs_sq']/(nimg*full_image_npixels)
            # sanity check that the truth base rate is the same for
            # each model tested -- if not, bug / different batches etc
            if baserate_first is None:
                baserate_first = baserate
            else:
                assert np.isclose(baserate, baserate_first)
            j = j + 1

        target_skill = 0.5 + baserate_first/2
        plt.semilogx([1.0, spatial_scales[-1]], [baserate, baserate],
                     '-', color='0.9', lw=lw)
        plt.semilogx([1.0, spatial_scales[-1]], [target_skill, target_skill],
                     '-', color='0.8', lw=lw)
        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
        # plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Spatial scale (km)')
        plt.ylabel('Fractions skill score (FSS)')
        plt.title(f'FSS curve for precip threshold {precip_values[i]}')
        plt.legend(loc="best")
        pltsave = os.path.join(log_path, f"FSS-{precip_values[i]}.pdf")
        plt.savefig(pltsave, bbox_inches='tight')
        plt.close()


def plot_rapsd(rapsd_files,
               num_samples,
               labels,
               log_path,
               spatial_scales):

    rapsd_data = {}
    for (fn, label) in zip(rapsd_files, labels):
        with open(fn, 'rb') as handle:
            rapsd_data[label] = pickle.load(handle)

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours += ['#56B4E9', '#009E73', '#F0E442']  # add some
    for k in range(num_samples):
        for model in labels:
            # sniff data labels
            rapsd_data_labels = list(rapsd_data[model][0].keys())
            fig, ax = plt.subplots()
            for (i, colour) in zip(range(len(rapsd_data_labels)), colours):
                if rapsd_data_labels[i] == 'IFS':
                    # skip the input data b/c the resolution is different
                    pass
                elif len(rapsd_data[model][k][rapsd_data_labels[i]].shape) != 2:
                    # if preds weren't made for this method don't plot
                    # if preds were made shape will be (940, 940)
                    pass
                else:
                    if rapsd_data_labels[i] == 'TRUTH':
                        lw = 2.0
                    else:
                        lw = 1.0
                    R_1, freq_1 = rapsd(rapsd_data[model][k][rapsd_data_labels[i]],
                                        fft_method=np.fft, return_freq=True)
                    # Plot the observed power spectrum and the model
                    plot_spectrum1d(freq_1,
                                    R_1,
                                    x_units="km",
                                    y_units="dBR",
                                    color=colour,
                                    ax=ax,
                                    lw=lw,
                                    label=rapsd_data_labels[i],
                                    wavelength_ticks=spatial_scales)
            ax.set_title(f"Radially averaged log-power spectrum - {k+1}")
            ax.legend(loc="best")
            pltsave = os.path.join(log_path, f"RAPSD-{model}-{k+1}.pdf")
            plt.savefig(pltsave, bbox_inches='tight')
            plt.close()


def plot_preds(pred_files,
               num_samples,
               labels,
               log_path,
               preds_to_plot,
               value_range_precip=(0.1, 30),
               palette="YlGnBu"):
    pred_data = {}
    for (fn, label) in zip(pred_files, labels):
        with open(fn, 'rb') as handle:
            pred_data[label] = pickle.load(handle)
    num_cols = num_samples
    num_rows = len(preds_to_plot)
    plt.figure(figsize=(1.5*num_cols, 1.5*num_rows), dpi=300)
    linewidth = 0.4
    cmap = ListedColormap(sns.color_palette(palette, 256))
    cmap.set_under('white')
    extent = [-7.5, 2, 49.5, 59]  # (lon, lat)
    alpha = 0.8
    spacing = 10

    # colorbar
    units = "Rain rate [mm h$^{-1}$]"
    cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 15, 30, 50])
    cb_tick_labels = [0.1, 0.5, 1, 2, 5, 15, 30, 50]

    gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols, wspace=0.5, hspace=0.5)

    for k in range(num_samples):
        for i in range(num_rows):
            if i < (num_rows)/2:
                label = 'GAN'
            else:
                label = 'VAEGAN'
            plt.subplot(gs[(spacing*i):(spacing+spacing*i), (spacing*k):(spacing+spacing*k)],
                        projection=ccrs.PlateCarree())
            ax = plt.gca()
            ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
            if i < (len(preds_to_plot)):
                plot_img_log_coastlines(pred_data[label][k][preds_to_plot[i]],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if i == 0:
                title = pred_data[label][k]['dates'][:4] + '-' + pred_data[label][k]['dates'][4:6] + '-' + pred_data[label][k]['dates'][6:8] + ' ' + str(pred_data[label][k]['hours']) + 'Z'
                plt.title(title, fontsize=9)

            if k == 0:
                if i < (len(preds_to_plot)):
                    ax.set_ylabel(preds_to_plot[i], fontsize=8)  # cartopy takes over the xlabel and ylabel
                    ax.set_yticks([])  # this weird hack restores them. WHY?!?!

    plt.suptitle('Example predictions for different input conditions')

    cax = plt.subplot(gs[-1, 1:-1]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), cmap=cmap, orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=12)
    cb.set_label(units, size=12)
    # cannot save as pdf - will produce artefacts
    plt.savefig("{}/predictions-{}.png".format(log_path,
                                               num_samples), bbox_inches='tight')
    plt.close()


def plot_comparison(files,
                    num_samples,
                    labels,
                    log_path,
                    comp_to_plot,
                    value_range_precip=(0.1, 30),
                    palette="mako_r"):

    pred_data = {}
    for (fn, label) in zip(files, labels):
        with open(fn, 'rb') as handle:
            pred_data[label] = pickle.load(handle)

    num_cols = num_samples
    num_rows = len(comp_to_plot)
    plt.figure(figsize=(1.5*num_cols, 1.5*num_rows), dpi=300)
    linewidth = 0.4
    cmap = ListedColormap(sns.color_palette(palette, 256))
    cmap.set_under('white')
    extent = [-7.5, 2, 49.5, 59]  # (lon, lat)
    alpha = 0.8
    spacing = 10

    # colorbar
    units = "Rain rate [mm h$^{-1}$]"
    cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 15, 30, 50])
    cb_tick_labels = [0.1, 0.5, 1, 2, 5, 15, 30, 50]

    gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols, wspace=0.5, hspace=0.5)

    for k in range(num_samples):
        for i in range(num_rows):
            plt.subplot(gs[(spacing*i):(spacing+spacing*i), (spacing*k):(spacing+spacing*k)],
                        projection=ccrs.PlateCarree())
            ax = plt.gca()
            ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
            if i == 0:  # plot IFS
                plot_img_log_coastlines(pred_data['4x'][k]['IFS'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
                title = pred_data['4x'][k]['dates'][:4] + '-' + pred_data['4x'][k]['dates'][4:6] + '-' + pred_data['4x'][k]['dates'][6:8] + ' ' + str(pred_data['4x'][k]['hours']) + 'Z'
                plt.title(title, fontsize=9)
            if i == 1:  # plot TRUTH
                plot_img_log_coastlines(pred_data['4x'][k]['TRUTH'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if i > 1:  # plot different flavours of prediction
                plot_img_log_coastlines(pred_data[comp_to_plot[i]][k]['GAN pred 1'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if k == 0:
                if i < (len(comp_to_plot)):
                    ax.set_ylabel(comp_to_plot[i], fontsize=8)  # cartopy takes over the xlabel and ylabel
                    ax.set_yticks([])  # this weird hack restores them. WHY?!?!

    plt.suptitle('Example predictions for different input conditions')

    cax = plt.subplot(gs[-1, 1:-1]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), cmap=cmap, orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=12)
    cb.set_label(units, size=12)
    # cannot save as pdf - will produce artefacts
    plt.savefig("{}/comparison-{}.png".format(log_path,
                                              num_samples), bbox_inches='tight')
    plt.close()
