# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_fcst_fields

import os
import sys
import pathlib
import yaml

import xarray as xr
import numpy as np
from tensorflow.keras.utils import Progbar

sys.path.insert(1, "../")
from data.data_gefs import (
    HOURS,
    all_fcst_fields,
    nonnegative_fields,
    fcst_norm,
    logprec,
    denormalise,
    load_hires_constants,
)
from config import set_gpu_mode, get_data_paths, read_downscaling_factor
from setupmodel import setup_model
from model.noise import NoiseGenerator
import tensorflow as tf

from datetime import datetime, timedelta

# %%
# Define the latitude and longitude arrays for later
latitude  = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15,  54.3, 0.1)

# Group A: single file per day containing ALL time steps (time dim = 37)
GROUP_A = ['evaporation', 'precipitation_cumulative_mean']

# Group B: one file per time step, valid hours are 00 / 06 / 12 / 18
#          all files stay in the init-day folder regardless of lead time
GROUP_B = [
    'specific_cloud_ice_water_content_500',
    'specific_cloud_ice_water_content_700',
    'specific_cloud_ice_water_content_850',
    'u_component_of_wind_500',
    'u_component_of_wind_700',
    'u_component_of_wind_850',
    'v_component_of_wind_500',
    'v_component_of_wind_700',
    'v_component_of_wind_850'
]

# Some setup
set_gpu_mode()
data_paths        = get_data_paths()
downscaling_steps = read_downscaling_factor()["steps"]
assert fcst_norm is not None

# %%
# Open and parse forecast.yaml
fcstyaml_path = "../config/forecast_gfs.yaml"
with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# %%
model_folder     = fcst_params["MODEL"]["folder"]
checkpoint       = fcst_params["MODEL"]["checkpoint"]
input_folder     = fcst_params["INPUT"]["folder"]
dates            = fcst_params["INPUT"]["dates"]
start_hour       = fcst_params["INPUT"]["start_hour"]
end_hour         = fcst_params["INPUT"]["end_hour"]
output_folder    = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]

assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
assert end_hour   % HOURS == 0, f"end_hour must be divisible by {HOURS}"

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode             = setup_params["GENERAL"]["mode"]
arch             = setup_params["MODEL"]["architecture"]
padding          = setup_params["MODEL"]["padding"]
filters_gen      = setup_params["GENERATOR"]["filters_gen"]
noise_channels   = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc     = setup_params["DISCRIMINATOR"]["filters_disc"]
constant_fields  = 2

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN — 1 channel per field (no ensemble mean/std)
weights_fn     = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = len(all_fcst_fields)   # = 11  (1 canal par field)

model = setup_model(
    mode=mode,
    arch=arch,
    downscaling_steps=downscaling_steps,
    input_channels=input_channels,
    constant_fields=constant_fields,
    filters_gen=filters_gen,
    filters_disc=filters_disc,
    noise_channels=noise_channels,
    latent_variables=latent_variables,
    padding=padding
)
gen = model.gen
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1)  # 1 x lats x lons x 2


# %%
def load_and_interpolate_field(field, d, in_time_idx, input_folder,
                                target_lat, target_lon):
    """
    Load a NGCM field for a given day and time step,
    then interpolate it onto the GAN model target grid.

    Parameters
    ----------
    field         : str      — field name
    d             : datetime — forecast init date
    in_time_idx   : int      — time step index (0→00h, 1→06h, ...)
    input_folder  : str      — root NGCM folder, e.g. /content/drive/.../NGCM/
                               Structure: <input_folder>/<field>/<year>/<file>
    target_lat    : ndarray  — target latitude array
    target_lon    : ndarray  — target longitude array

    Returns
    -------
    data : ndarray of shape (len(target_lat), len(target_lon))
    """

    hour = in_time_idx * HOURS   # lead time: 0, 6, 12, 18, 24, 30 ...

    if field in GROUP_A:
        # Single file for all time steps → always the init-day 00h file
        file_hour = 0
    else:
        # One file per time step, all stored in the init-day folder
        # Hours cycle: 00 / 06 / 12 / 18 / 00 / ... via modulo 24
        file_hour = hour % 24

    input_file = (
        f"{field}_{d.year}_ngcm_{field}_2.8deg_6h_GHA"
        f"_{d.strftime('%Y%m%d')}_{file_hour:02d}h.nc"
    )

    # Structure: <input_folder>/<field>/<year>/<file>
    nc_in_path = os.path.join(input_folder, field, str(d.year), input_file)

    print(f"  → Loading: {nc_in_path}")

    if not os.path.exists(nc_in_path):
        raise FileNotFoundError(
            f"File not found for field='{field}', "
            f"init_date={d.strftime('%Y%m%d')}, file_hour={file_hour:02d}h\n"
            f"Path attempted: {nc_in_path}"
        )

    # Open file
    nc_file    = xr.open_dataset(nc_in_path, decode_timedelta=False)
    short_name = [v for v in nc_file.data_vars][0]
    data       = nc_file[short_name]

    # Extract correct time step for Group A
    if field in GROUP_A:
        data = data.isel(time=in_time_idx)
        if 'surface' in data.dims:
            data = data.squeeze("surface")

    # Convert to numpy — must be 2D
    data = data.values
    assert data.ndim == 2, (
        f"Unexpected shape for '{field}': {data.shape}, expected 2D"
    )

    # Check for missing values
    if np.all(np.isnan(data)):
        raise ValueError(
            f"All values are NaN for field='{field}', "
            f"init_date={d.strftime('%Y%m%d')}, file_hour={file_hour:02d}h"
        )

    # Interpolate onto target grid
    lat_dim = "latitude" if "latitude" in nc_file.coords else "lat"
    lon_dim = "longitude" if "longitude" in nc_file.coords else "lon"

    da = xr.DataArray(
        data,
        dims=[lon_dim, lat_dim],
        coords={
            lon_dim: nc_file[lon_dim].values,
            lat_dim: nc_file[lat_dim].values,
        }
    )
    data_interp = da.interp(
        {lat_dim: target_lat, lon_dim: target_lon},
        method="linear"
    ).values

    # Guarantee output shape is (lat, lon)
    if data_interp.shape == (len(target_lon), len(target_lat)):
        data_interp = data_interp.T

    print(f"  → Shape after interpolation: {data_interp.shape}")
    nc_file.close()

    return data_interp   # (len(target_lat), len(target_lon))


# %%
def make_fcst(input_folder=input_folder, output_folder=output_folder,
              dates=dates, start_hour=start_hour, end_hour=end_hour, HOURS=HOURS,
              all_fst_fields=all_fcst_fields, nonnegative_fields=nonnegative_fields,
              gen=gen, ensemble_members=ensemble_members):

    dates       = np.asarray(dates, dtype='datetime64[ns]')
    valid_times = np.arange(start_hour, end_hour + 1, HOURS)

    for day in dates:
        d = datetime(
            day.astype('datetime64[D]').astype(object).year,
            day.astype('datetime64[D]').astype(object).month,
            day.astype('datetime64[D]').astype(object).day
        )

        print(f"{d.year}-{d.month:02}-{d.day:02}")

        # Specify input folder for year
        input_folder_year = input_folder + f"{d.year}/"

        # Create output folder
        output_folder_year = output_folder + f"test/{d.year}/"
        pathlib.Path(output_folder_year).mkdir(parents=True, exist_ok=True)

        # Pre-allocate output array: (members, valid_times, lat, lon)
        n_valid_times = len(range(start_hour // HOURS, end_hour // HOURS))
        precip_all = np.full(
            (ensemble_members, n_valid_times, len(latitude), len(longitude)),
            fill_value=np.nan,
            dtype=np.float32
        )

        for out_time_idx, in_time_idx in enumerate(
            range(start_hour // HOURS, end_hour // HOURS)
        ):
            d_valid = d + timedelta(hours=int(in_time_idx * HOURS))
            day_idx = (int(in_time_idx * HOURS) // 24) - 1

            field_arrays = []

            for field in all_fcst_fields:

                # Load and interpolate → shape (lat, lon)
                data = load_and_interpolate_field(
                    field=field,
                    d=d,
                    in_time_idx=in_time_idx,
                    input_folder=input_folder,
                    target_lat=latitude,
                    target_lon=longitude
                )

                if field in nonnegative_fields:
                    data = np.maximum(data, 0.0)

                if field in ["precipitation_cumulative_mean"]:
                    data = np.log10(1 + data)

                if field in ["evaporation"]:
                    if fcst_norm[field]["max"] != 0:
                        data = data / fcst_norm[field]["max"]

                elif 'specific_cloud_ice_water_content' in field:
                    if fcst_norm[field]["max"] != 0:
                        data = data / fcst_norm[field]["max"]

                elif 'u_component_of_wind' in field or 'v_component_of_wind' in field:
                    norm_val = max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
                    if norm_val != 0:
                        data = data / norm_val

                field_arrays.append(data[..., np.newaxis])  # (lat, lon, 1)

            network_fcst_input = np.concatenate(field_arrays, axis=-1)         # lat x lon x 11
            network_fcst_input = np.expand_dims(network_fcst_input, axis=0)    # 1 x lat x lon x 11
            noise_shape        = network_fcst_input.shape[1:-1] + (noise_channels,)
            noise_gen          = NoiseGenerator(noise_shape, batch_size=1)

            progbar = Progbar(ensemble_members)
            for ii in range(ensemble_members):
                gan_inputs     = [network_fcst_input, network_const_input, noise_gen()]
                gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
                precip_all[ii, out_time_idx, :, :] = \
                    denormalise(gan_prediction[0, :, :, 0]).astype(np.float32)
                progbar.add(1)

        # Save as .npz
        # Reload with:
        #   data   = np.load("GAN_20180101.npz")
        #   precip = data["precipitation"]  → (members, times, lat, lon)
        #   lats   = data["latitude"]
        #   lons   = data["longitude"]
        #   vtimes = data["valid_times"]    → lead hours [0, 6, 12, ...]
        npz_out_path = os.path.join(
            output_folder_year, f"GAN_{d.year}{d.month:02}{d.day:02}.npz"
        )
        np.savez_compressed(
            npz_out_path,
            precipitation = precip_all,
            latitude      = latitude.astype(np.float32),
            longitude     = longitude.astype(np.float32),
            valid_times   = valid_times[:n_valid_times].astype(np.float32),
        )
        print(f"Saved: {npz_out_path}  shape={precip_all.shape}")


if __name__ == "__main__":
    make_fcst()
