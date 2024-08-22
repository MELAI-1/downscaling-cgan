# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

HOME = Path(os.getcwd()).parents[0]
sys.path.insert(1, str(HOME / 'downscaling-cgan'))
sys.path.insert(1, str(HOME))
sys.path.insert(1, str(HOME.parents[0]))

from dsrnngan.data.data import load_hires_constants, make_dataset_consistent, filter_by_lat_lon, load_land_sea_mask
from dsrnngan.evaluation.plots import lake_feature, border_feature, disputed_border_feature
from dsrnngan.utils.utils import special_areas

# %%
special_areas = copy.deepcopy(special_areas)

# %%
special_areas = {k:v for k,v in special_areas.items() if k in ['kenya', 'lake_victoria']}

# %%
lat_range_list = list(np.arange(-11.95, 15.05, 0.1))
lon_range_list = list(np.arange(25.05, 51.35, 0.1))

# %%
oro_ds = xr.load_dataset('/network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_data/h_HRES_EAfrica.nc')
oro_ds = filter_by_lat_lon(oro_ds, lon_range=lon_range_list, lat_range=lat_range_list)
oro_ds = make_dataset_consistent(oro_ds)
h_vals = oro_ds['h'].values
h_vals[h_vals < 2] = 2


# %%
from matplotlib import colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap    

cmap = plt.get_cmap('terrain')
new_cmap = truncate_colormap(cmap, 0.2, 0.9)

# %%
special_areas

# %%
from matplotlib import gridspec

plt.rcParams.update({'font.size': 20})


from cartopy.feature import LAKES, COLORS, OCEAN

lake_feature = cfeature.NaturalEarthFeature(
    'physical', 'lakes',
    cfeature.auto_scaler, edgecolor='black', facecolor=COLORS['water'])
ocean_feature = cfeature.NaturalEarthFeature(
    'physical', 'ocean',
    cfeature.auto_scaler, edgecolor='black', facecolor=COLORS['water'])

# fig = plt.figure(constrained_layout=True, figsize=(12, 10))
# gs = gridspec.GridSpec(1, 2, figure=fig, 
#                         width_ratios=[1, 0.05],
#                         wspace=0.005)    
# ax = fig.add_subplot(gs[0,0], projection = ccrs.PlateCarree())
fig, ax = plt.subplots(1,1, subplot_kw={'projection' : ccrs.PlateCarree()}, figsize=(10,10))
ax.set_extent([min(lon_range_list), max(lon_range_list), min(lat_range_list), max(lat_range_list)])
# ax.background_img('vhigh')
hlevels = levels=[0, 10,50, 100] + list(np.arange(250, 3000, 250))
for k, sa_dict in special_areas.items():
    
    im = ax.contourf(lon_range_list, lat_range_list, h_vals[0,:,:], transform=ccrs.PlateCarree(), cmap=new_cmap, 
                levels=hlevels, extend='max')#
    ax.coastlines(resolution='10m', color='black', linewidth=0.4)
    ax.add_feature(border_feature)
    # ax.add_feature(lake_feature, alpha=1)
    ax.add_feature(lake_feature)
    ax.add_feature(disputed_border_feature)
    ax.add_feature(ocean_feature)
    ln_rng, lt_rng = sa_dict['lon_range'], sa_dict['lat_range']

    ax.plot([ln_rng[0], ln_rng[0]], [lt_rng[0], lt_rng[1]], color='r', linestyle='--', linewidth=2,transform=ccrs.PlateCarree())
    ax.plot([ln_rng[1], ln_rng[1]], [lt_rng[0], lt_rng[1]], color='r', linestyle='--', linewidth=2,transform=ccrs.PlateCarree())
    ax.plot([ln_rng[0], ln_rng[1]], [lt_rng[0], lt_rng[0]], color='r', linestyle='--', linewidth=2,transform=ccrs.PlateCarree())
    ax.plot([ln_rng[0], ln_rng[1]], [lt_rng[1], lt_rng[1]], color='r', linestyle='--', linewidth=2,transform=ccrs.PlateCarree(), label=k)
    ax.text(ln_rng[0], lt_rng[1] + 0.15, sa_dict['abbrv'], fontsize=12, color='k', bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
# ax.background_img(extent=[25, 50, -12,10])
# cbar_ax = fig.add_subplot(gs[:,-1])
# cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', shrink = 0.1, aspect=2)
cbar = plt.colorbar(im,values=hlevels,pad=0.01, shrink = 0.8)
cbar.set_label("Orography (m)", loc='center')

lon_ticks = np.arange(25, 50,10)
lat_ticks = np.arange(-10, 15,10)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)
ax.set_xticklabels([f'{ln}E' for ln in lon_ticks])
ax.set_yticklabels([f"{np.abs(lt)}{'N' if lt >=0 else 'S'}" for lt in lat_ticks])

plt.savefig('/network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_plots/area_range.pdf', format='pdf', bbox_inches='tight')

# %%
special_areas['lake_victoria']

# %%
# Calculate lake victoria area vs land area

lsm = load_land_sea_mask(
                       latitude_vals=latitude_range, 
                       longitude_vals=longitude_range,
                       filepath='/network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_data/lsm_HRES_EAfrica.nc')

# %%
lsm

# %%
sa_dict = special_areas['lake_victoria']

latitude_vals = np.arange(sa_dict['lat_range'][0], sa_dict['lat_range'][1], 0.1)
longitude_vals = np.arange(sa_dict['lon_range'][0], sa_dict['lon_range'][1], 0.1)

lsm = load_land_sea_mask(
                       latitude_vals=latitude_vals, 
                       longitude_vals=longitude_vals,
                       filepath='/network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_data/lsm_HRES_EAfrica.nc')
# lsm.sel(latitude=slice(sa_dict['lat_range'][0], sa_dict['lat_range'][1])).sel(longitude=slice(sa_dict['lon_range'][0], sa_dict['lon_range'][1]))

# %%
def get_area_from_shape(s):

    return s[0]*s[1]

get_area_from_shape(lsm.shape)

# %%
lsm[lsm==0].shape

# %%
print(f'Full LV area = {get_area_from_shape(lsm.shape)} grid squares')
print(f'Lake area = {len(lsm[lsm<=0.5].flatten())} grid squares')
print(f'Lake area = {len(lsm[lsm>0.5].flatten())} grid squares')

# %%

fig, ax = plt.subplots(1,1, figsize=(8,8))
plt.imshow(lsm, origin='lower')

# %%

# %%
