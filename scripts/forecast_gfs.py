# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_ngcm_fields

import os
import sys
import shutil
import pathlib
import yaml
import gcsfs
from pathlib import Path

import netCDF4 as nc
from cftime import date2num
import xarray as xr
import numpy as np
from tensorflow.keras.utils import Progbar

sys.path.insert(1, "../")
from dsrnngan.data.data_gefs import (
    HOURS,
    all_ngcm_fields,
    nonnegative_fields,
    log_plus_1,
    normalise_precipitation,
    denormalise,
    load_hires_constants,
    load_ngcm_stats,
    interpolate_dataset_on_lat_lon,
    OROGRAPHY_PATH,
    LSM_PATH
)

from dsrnngan.utils.read_config import set_gpu_mode, get_data_paths
from dsrnngan.model.setupmodel import setup_model
from dsrnngan.model.noise import NoiseGenerator
import tensorflow as tf
from dsrnngan.utils.read_config import read_model_config, read_data_config

from datetime import datetime, timedelta

# =============================================================================
# GRID DEFINITION
# Use linspace instead of arange to avoid floating point instability
# =============================================================================
latitude  = np.linspace(-13.65, 24.65, 384)
longitude = np.linspace(19.15,  54.25, 352)

# Group A: files contain all time steps in a single file (time dim = 37)
GROUP_A = ['evaporation', 'precipitation_cumulative_mean']

# Group B: one file per time step, no time dimension
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

# =============================================================================
# SETUP
# =============================================================================
set_gpu_mode()
data_paths = get_data_paths()

fcst_norm = load_ngcm_stats()
assert fcst_norm is not None, "Could not load forecast normalisation stats"

# =============================================================================
# LOAD FORECAST YAML CONFIG
# =============================================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
fcstyaml_path = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "config", "forecast_gfs.yaml")
)

with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

model_folder      = fcst_params["MODEL"]["folder"]
checkpoint        = fcst_params["MODEL"]["checkpoint"]
input_folder      = fcst_params["INPUT"]["folder"]
dates             = fcst_params["INPUT"]["dates"]
start_hour        = fcst_params["INPUT"]["start_hour"]
end_hour          = fcst_params["INPUT"]["end_hour"]
output_folder     = fcst_params["OUTPUT"]["folder"]
ensemble_members  = fcst_params["OUTPUT"]["ensemble_members"]
config_path       = fcst_params["MODEL"]["config_path"]
MODEL_CONFIG_PATH = config_path
DATA_CONFIG_PATH  = fcst_params["Data"]["data_path"]

assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
assert end_hour   % HOURS == 0, f"end_hour must be divisible by {HOURS}"

# =============================================================================
# LOAD MODEL AND GAN CONFIGS
# =============================================================================
def load_yaml(path: str):
    """Load a YAML file from local filesystem or GCS."""
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        if not fs.exists(path):
            raise FileNotFoundError(f"{path} not found on GCS")
        with fs.open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{path} not found locally")
        with open(p, "r") as f:
            return yaml.safe_load(f)


# Connect to GCS and load setup params
fs = gcsfs.GCSFileSystem()
with fs.open(config_path, "r") as f:
    setup_params = yaml.safe_load(f)
print(setup_params)

mode             = setup_params["mode"]
arch             = setup_params["architecture"]
padding          = setup_params["padding"]
filters_gen      = setup_params["generator"]["filters_gen"]
noise_channels   = setup_params["generator"]["noise_channels"]
latent_variables = setup_params["generator"]["latent_variables"]
filters_disc     = setup_params["discriminator"]["filters_disc"]

# 1 channel per field — no ensemble in input data
input_channels = len(all_ngcm_fields)  # = 11

# Load model and data configs
model_cfg_dict = load_yaml(MODEL_CONFIG_PATH)
data_cfg_dict  = load_yaml(DATA_CONFIG_PATH)

model_config = read_model_config(model_config_dict=model_cfg_dict)
data_config  = read_data_config(data_config_dict=data_cfg_dict)

# =============================================================================
# SETUP GAN MODEL AND LOAD WEIGHTS
# =============================================================================
model = setup_model(
    model_config=model_config,
    data_config=data_config
)
gen = model.gen


def get_weights_path(params, repo_name="downscaling-cgan"):
    """Build the path to the generator weights file."""
    base_path   = Path(f"/content/{repo_name}")
    checkpoint  = params['MODEL']['checkpoint']
    rel_path    = params['MODEL'].get('local_weights_dir', 'notebooks/models')
    weight_file = f"gen_weights-{checkpoint:07d}.h5"
    full_path   = base_path / rel_path / weight_file
    if not full_path.exists():
        raise FileNotFoundError(f"Weights not found at: {full_path}")
    return str(full_path)


weights_fn = get_weights_path(fcst_params)
print(f"Model weights localized: {weights_fn}")
gen.load_weights(weights_fn)
print(f"Model successfully loaded")

# =============================================================================
# LOAD CONSTANT FIELDS (orography + land-sea mask)
# =============================================================================
data_paths = {"LSM": LSM_PATH, "OROGRAPHY": OROGRAPHY_PATH}
network_const_input = load_hires_constants(
    batch_size=1,
    fields=["orography", "lsm"],
    data_paths=data_paths
)  # shape: 1 x lats x lons x 2

# =============================================================================
# OUTPUT FILE CREATION
# =============================================================================
def create_output_file(nc_out_path):
    """Create the output NetCDF file and return a dict of its variables."""

    # Delete file if it already exists to avoid HDF lock error
    if os.path.exists(nc_out_path):
        os.remove(nc_out_path)
        print(f"  → Removed existing file: {nc_out_path}")

    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 6-hour rainfall ensemble members in the ICPAC region."

    # Dimensions
    rootgrp.createDimension("latitude",   len(latitude))
    rootgrp.createDimension("longitude",  len(longitude))
    rootgrp.createDimension("member",     ensemble_members)
    rootgrp.createDimension("time",       None)
    rootgrp.createDimension("valid_time", None)

    # Coordinate variables
    latitude_data = rootgrp.createVariable("latitude", "f4", ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude

    longitude_data = rootgrp.createVariable("longitude", "f4", ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude

    ensemble_data = rootgrp.createVariable("member", "i4", ("member",))
    ensemble_data.units = "ensemble member"
    ensemble_data[:] = range(1, ensemble_members + 1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time", "f4", ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable(
        "fcst_valid_time", "f4", ("time", "valid_time")
    )
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["precipitation"] = rootgrp.createVariable(
        "precipitation", "f4",
        ("time", "member", "valid_time", "latitude", "longitude"),
        compression="zlib",
        chunksizes=(1, 1, 1, len(latitude), len(longitude))
    )
    netcdf_dict["precipitation"].units     = "mm h**-1"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    return netcdf_dict


# =============================================================================
# FIELD LOADING AND INTERPOLATION
# =============================================================================
def load_and_interpolate_field(field, d, in_time_idx, input_folder_year,
                                target_lat, target_lon):
    """
    Load a NGCM field for a given day and time step,
    then interpolate it onto the GAN model target grid.

    Parameters:
    -----------
    field             : field name (e.g. 'evaporation', 'u_component_of_wind_500')
    d                 : datetime of the day (e.g. datetime(2018, 1, 1))
    in_time_idx       : time step index (0→00h, 1→06h, 2→12h ...)
    input_folder_year : root folder for the year (e.g. '.../NGCM/2018/')
    target_lat        : target latitude array  (384,)
    target_lon        : target longitude array (352,)

    Returns:
    --------
    data_interp : numpy array of shape (384, 352)
    """

    # --- Step 1: build the file path ---
    hour = in_time_idx * HOURS  # 0→00h, 1→06h, 2→12h ...

    # Group A: all time steps in one file → always load the 00h file
    # Group B: one file per time step → use the correct hour
    file_hour  = 0 if field in GROUP_A else hour
    input_file = (
        f"{field}_{d.year}_ngcm_{field}_2.8deg_6h_GHA"
        f"_{d.strftime('%Y%m%d')}_{file_hour:02d}h.nc"
    )
    nc_in_path = os.path.join(input_folder_year, field, str(d.year), input_file)

    if not os.path.exists(nc_in_path):
        raise FileNotFoundError(
            f"File not found for field='{field}', "
            f"date={d.strftime('%Y%m%d')}, hour={file_hour:02d}h\n"
            f"Path attempted: {nc_in_path}"
        )
    print(f"  → Loading: {nc_in_path}")

    # --- Step 2: open the file ---
    nc_file    = xr.open_dataset(nc_in_path, decode_timedelta=False)
    short_name = [v for v in nc_file.data_vars][0]
    data       = nc_file[short_name]

    # --- Step 3: extract data according to group ---
    if field in GROUP_A:
        # time dimension has 37 steps → select the correct one
        data = data.isel(time=in_time_idx)
        # remove surface dimension if present
        if 'surface' in data.dims:
            data = data.squeeze("surface")
    # Group B: already shape (longitude: 16, latitude: 18), nothing to do

    # convert to numpy and verify shape
    data = data.values  # (16, 18)
    assert data.ndim == 2, (
        f"Unexpected shape for '{field}': {data.shape}, "
        f"expected 2D (longitude, latitude)"
    )

    # --- Step 4: check for missing values ---
    if np.all(np.isnan(data)):
        raise ValueError(
            f"All values are NaN for field='{field}', "
            f"date={d.strftime('%Y%m%d')}, hour={hour:02d}h"
        )

    # --- Step 5: interpolate onto the target grid ---
    da = xr.DataArray(
        data,
        dims=["longitude", "latitude"],
        coords={
            "longitude": nc_file.longitude.values,  # (16,): 16.88 ... 59.06
            "latitude":  nc_file.latitude.values     # (18,): -18.14 ... 29.3
        }
    )
    data_interp = da.interp(
        latitude=target_lat,   # (384,)
        longitude=target_lon,  # (352,)
        method="linear"
    ).values  # shape (352, 384) or (384, 352) → verified on first call

    print(f"  → Shape after interpolation: {data_interp.shape}")

    nc_file.close()
    return data_interp


# =============================================================================
# MAIN FORECAST FUNCTION
# =============================================================================
def make_fcst(input_folder=input_folder, output_folder=output_folder,
              dates=dates, start_hour=start_hour, end_hour=end_hour,
              HOURS=HOURS, all_fst_fields=all_ngcm_fields,
              nonnegative_fields=nonnegative_fields,
              gen=gen, ensemble_members=ensemble_members):

    dates       = np.asarray(dates, dtype='datetime64[ns]')
    valid_times = np.arange(start_hour, end_hour + 1, HOURS)

    for day in dates:
        d = datetime(
            day.astype('datetime64[D]').astype(object).year,
            day.astype('datetime64[D]').astype(object).month,
            day.astype('datetime64[D]').astype(object).day
        )
        print(f"\n{'='*60}")
        print(f"Processing date: {d.year}-{d.month:02}-{d.day:02}")
        print(f"{'='*60}")

        input_folder_year = os.path.join(input_folder, str(d.year))

        # Final output path on Drive
        output_folder_year = os.path.join(output_folder, "test", str(d.year))
        pathlib.Path(output_folder_year).mkdir(parents=True, exist_ok=True)
        nc_out_path_drive = os.path.join(
            output_folder_year, f"GAN_{d.year}{d.month:02}{d.day:02}.nc"
        )

        # Local temp path — avoids HDF error when writing directly to Drive
        local_tmp_dir = f"/content/tmp_output/test/{d.year}/"
        pathlib.Path(local_tmp_dir).mkdir(parents=True, exist_ok=True)
        nc_out_path_local = os.path.join(
            local_tmp_dir, f"GAN_{d.year}{d.month:02}{d.day:02}.nc"
        )

        # # Write NetCDF to local path
        # netcdf_dict = create_output_file(nc_out_path_local)
        # netcdf_dict["time_data"][0] = date2num(
        #     d, units="hours since 1900-01-01 00:00:00.0"
        # )

        for out_time_idx, in_time_idx in enumerate(
            range(start_hour // HOURS, end_hour // HOURS)
        ):
            hour = in_time_idx * HOURS
            print(f"\n  Time step {in_time_idx} → {hour:02d}h")

            # # Write valid time
            # netcdf_dict["valid_time_data"][0, out_time_idx] = date2num(
            #     d + timedelta(hours=int(valid_times[out_time_idx])),
            #     units="hours since 1900-01-01 00:00:00.0"
            # )

            field_arrays = []

            # -----------------------------------------------------------------
            # Loop over all input fields
            # -----------------------------------------------------------------
            for field in all_ngcm_fields:

                # Load and interpolate field onto model grid
                data = load_and_interpolate_field(
                    field=field,
                    d=d,
                    in_time_idx=in_time_idx,
                    input_folder_year=input_folder_year,
                    target_lat=latitude,
                    target_lon=longitude
                )

                # Clip negative values for non-negative fields
                if field in nonnegative_fields:
                    data = np.maximum(data, 0.0)

                # Normalise according to field type
                if field == 'precipitation_cumulative_mean':
                    data = np.log10(1 + data)

                elif field == 'evaporation':
                    if fcst_norm[field]["max"] != 0:
                        data = data / fcst_norm[field]["max"]

                elif 'specific_cloud_ice_water_content' in field:
                    if fcst_norm[field]["max"] != 0:
                        data = data / fcst_norm[field]["max"]

                elif 'u_component_of_wind' in field or 'v_component_of_wind' in field:
                    norm_val = max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
                    if norm_val != 0:
                        data = data / norm_val

                # Add channel dimension → (384, 352, 1)
                data = data[..., np.newaxis]
                field_arrays.append(data)

            # -----------------------------------------------------------------
            # Concatenate all fields → (384, 352, 11)
            # -----------------------------------------------------------------
            network_fcst_input = np.concatenate(field_arrays, axis=-1)
            print(f"\n  network_fcst_input shape: {network_fcst_input.shape}")

            # Add batch dimension → (1, 384, 352, 11)
            network_fcst_input = np.expand_dims(network_fcst_input, axis=0)

            # Build noise generator
            noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
            noise_gen   = NoiseGenerator(noise_shape, batch_size=1)

            print(f"  fcst input : {network_fcst_input.shape}")
            print(f"  const input: {network_const_input.shape}")
            print(f"  noise      : {noise_gen().shape}")

            # -----------------------------------------------------------------
            # Generate ensemble members via GAN
            # Each call to noise_gen() produces a different noise → different member
            # -----------------------------------------------------------------
            progbar = Progbar(ensemble_members)
            for ii in range(ensemble_members):
                gan_inputs     = [network_fcst_input, network_const_input, noise_gen()]
                gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
                netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = \
                    denormalise(gan_prediction[0, :, :, 0])
                progbar.add(1)

        # # Close local file
        # netcdf_dict["rootgrp"].close()
        # print(f"\n  Local file written: {nc_out_path_local}")

        # # Copy from local to Google Drive
        # shutil.copy(nc_out_path_local, nc_out_path_drive)
        # print(f"  Copied to Drive: {nc_out_path_drive}")

        # # Clean up local temp file
        # os.remove(nc_out_path_local)
        # print(f"  Local temp file removed")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    make_fcst()
