# Downscaling conditional Generative Adversarial Network (cGAN)

Code to downscale or postprocess NWP forecasts using a cGAN. Forked from the code at [https://github.com/bobbyantonio/downscaling-cgan].

Main modifications

<!-- Making the code more generally applicable to different regions, and different domain sizes.
- Making the data and models more config-driven.
- Adding extra features, like changes to input and output normalisation.
- Ce paragraphe explique la structure du projet -->
# Setup

Uses Python3.9 and Tensorflow 2.8.2

## To use with GPU (recommended) 

Create a new conda enviroment with:

`conda env create -f environment.yaml`


## To use with CPU

Create a new conda enviroment with:

`conda env create -f environment_cpu.yaml`

## Notes

Works on Linux, possibly not on Windows.

I found that there were often problems with conda accessing libraries, which was fixed by modifying the LD_LIBRARY_PATH environment variable. So if you are getting error messages that certain libraries (particularly things like c compiler libraries) aren't being accessed, then try:

`export LD_LIBRARY_PATH=PATH_TO_MINICONDA_FOLDER/lib/:$LD_LIBRARY_PATH`


[^1]: If numba is not available, we suggest you replace `from properscoring import crps_ensemble` to `from crps import crps_ensemble` in `evaluation.py` and `run_benchmarks.py`. This is because properscoring will fall back to an inefficient and memory-heavy CRPS implementation in this case.


# Preparing the data

You should have three datasets:
1. Forecast data (low-resolution)
2. Truth data, for example radar (high-resolution)
3. "Constant" data - orography and land-sea mask, at the same resolution as the truth data.

All images in each dataset should be the same size, and there should be a constant resolution scaling factor between them.  Enter this downscaling factor via the `downscaling_factor` parameter in `config/model_config.yaml`, along with a list of steps that multiply to the overall factor in `downscaling_steps`.  See `models.py` for exactly how these are used in the architecture.

If you do not want to use similar constant data, you will need to adjust the code and model architecture slightly.

Ideally these datasets are as 'clean' as possible.  We recommend you generate these with offline scripts that perform all regridding, cropping, interpolation, etc., so that the files can be loaded in and read directly with as little further processing as possible.  We saved these as netCDF files, which we read in using xarray.

We assume that it is possible to perform inference using full-size images, but that the images are too large to use whole during training.

For this reason, part of our dataflow involves generating training data separately (small portions of the full image), and storing these in .tfrecords files.  We think this has two advantages:
1. The training data is in an 'optimised' format; only the data needed for training is loaded in, rather than needing to open several files and extract a single timestep from each.
2. The training data can be split into different bins, and data can be drawn from these bins in a specific ratio.  We use this to artificially increase the prevalence of rainy data.

## Creating tfrecords for training

Tfrecords are created using the `write_data` function in `dsrnngan/data/tfrecords_generator.py`. This is called as: 


```bash
python -m dsrnngan.data.tfrecords_generator [-h] [--fcst-hours FCST_HOURS [FCST_HOURS ...]] [--records-folder RECORDS_FOLDER] [--data-config-path DATA_CONFIG_PATH]
               [--model-config-path MODEL_CONFIG_PATH] [--debug]

```
### Arguments

|short|long|default|help|
| :--- | :-------- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--fcst-hours`|`array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,        17, 18, 19, 20, 21, 22, 23])`|Hour(s) to process (space separated)|
||`--records-folder`|`None`|Root folder in which to create new tfrecords folder|
||`--data-config-path`|`None`|Path to data config yaml file|
||`--model-config-path`|`None`|Path to model config file (used for correct date ranges of training and validation sets)|
||`--debug`||Debug flag; use when debugging to reduce data sizes|

Start by adjusting the paths in `config/data_config.yaml` to point at your own datasets.  You can set up multiple configurations, e.g., if you are running your code on different machines that have data stored in different paths. Which set of paths is used is controlled by the `data_paths` parameter.

The file `dsrnngan/utils/read_config.py` controls what these options do, and you may wish to add more options of your own.

The functions in `dsrnngan/data/data.py` control how the forecast and observation data are loaded. You may need to rewrite substantial parts of these functions, according to the data that you plan to use.  As written, these functions are centred around the particular satellite data used for our study.  Essentially, for a particular date and time, the observation data for that time is loaded. A corresponding forecast is found (the logic for this is in `load_fcst()`), and that data is loaded.  You may wish to flip this logic around to be more forecast-centric.

A particular trap:
- Beware the difference between instanteneous fields (use `field[hour]`) and accumulated fields (use `field[hour] - field[hour-1]`).  Naturally, beware of off-by-one errors, etc., too.

These functions are ultimately called by the `DataGenerator` class in `dsrnngan/data/data_generator.py`.  This class represents a dataset of full-size images, which may be used for inference, or to create training data from.

At this point, you may wish to visualise the data returned by the `DataGenerator` to check that it matches what you expect!


The first time the data is prepared, statistics of the input forecats variables are gathered, and then cached for future runs (via the function get_ifs_stats in `dsrnngan/data/data.py`). This gives certain field statistics (mean, standard dev., max, etc.) that are used for normalising inputs to the neural networks during training. The NN performance should not be sensitive to the exact values of these, so it will be fine to run for just 1 year (among the training data years). Set the year to use for this normalisation via the `normalisation_year` parameter in the data config.

Next, you will want to manually run the function `write_data` in `tfrecords_generator.py`.  This generates training data by subsampling the full-size images. The typical command you will run is:

`python -m dsrnngan.data.tfrecords_generator --records-folder PATH_TO_ROOT_DATA_FOLDER --fcst-hours FCST_H --data-config-pathPATH_TO_DATA_CONFIG.yaml --model-config-path PATH_TO_MODEL_CONFIG`


The training data is separated into several bins/classes, according to what proportion of the image has rainfall.  You may wish to edit how this is performed!

Data config structure

# Training and evaluating a network

Models are trained and evaluated by running `dsrnngan/data/main.py`. This is called as: 


```bash
python -m dsrnngan.main [-h] [--records-folder RECORDS_FOLDER] [--model-folder MODEL_FOLDER | --model-config-path MODEL_CONFIG_PATH] [--no-train] [--restart] [--eval-model-numbers EVAL_MODEL_NUMBERS [EVAL_MODEL_NUMBERS ...] |
               --eval-full | --eval-short | --eval-blitz] [--num-samples NUM_SAMPLES] [--num-images NUM_IMAGES] [--eval-ensemble-size EVAL_ENSEMBLE_SIZE] [--eval-on-train-set] [--noise-factor NOISE_FACTOR]
               [--val-ym-start VAL_YM_START] [--val-ym-end VAL_YM_END] [--no-shuffle-eval] [--save-generated-samples] [--training-weights TRAINING_WEIGHTS TRAINING_WEIGHTS TRAINING_WEIGHTS TRAINING_WEIGHTS]
               [--output-suffix OUTPUT_SUFFIX] [--log-folder LOG_FOLDER] [--debug] [--wandb-logging]

```
## Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--records-folder`|`None`|Folder from which to gather the tensorflow records|
||`--model-folder`|`None`|Folder in which previous model configs and training results have been stored.|
||`--model-config-path`|`None`|Full path of model config yaml file to use.|
||`--no-train`||Do NOT carry out training, only perform eval|
||`--restart`||Restart training from latest checkpoint|
||`--eval-model-numbers`|`None`|Model number(s) to evaluate on (space separated)|
||`--eval-full`|`None`|Evaluate on all checkpoints.|
||`--eval-short`|`None`|Evaluate on last third of checkpoints|
||`--eval-blitz`|`None`|Evaluate on last 4 checkpoints.|
||`--num-samples`|`None`|Number of samples to train on (overrides value in config)|
||`--num-eval-images`|`20`|Number of images to evaluate on|
||`--eval-ensemble-size`|`None`|Size of ensemble to evaluate on|
||`--eval-on-train-set`||Use this flag to make the evaluation occur on the training dates|
||`--noise-factor`|`0.001`|Multiplicative noise factor for rank histogram|
||`--val-ym-start`|`None`|Validation start in YYYYMM format (defaults to range specified in config)|
||`--val-ym-end`|`None`|Validation start in YYYYMM format (defaults to the range specified in the config)|
||`--no-shuffle-eval`||Boolean, will turn off shuffling at evaluation.|
||`--save-generated-samples`||Flag to trigger saving of the evaluation arrays|
||`--training-weights`|`None`|Weighting of classes to use in training (assumes data has been split according to e.g. mean rainfall)|
||`--output-suffix`|`None`|Suffix to append to model folder. If none then model folder has same name as TF records folder used as input.|
||`--log-folder`|`None`|Root folder to which models are saved|
||`--wandb-loggin`||Turn on logging to weights and biases (requires account setup as detailed below)|
||`--debug`||Flag to trigger debug mode, to reduce data volumes|


Model parameters are set in the model configuration (.yaml) file.
An example is provided in the main directory. We recommend copying this 
file to somewhere on your local machine before training.

To restart training for a particular model, just specify the `--model-folder`; the program will read the configs from within this folder. Use `--restart` to make sure the training starts from the most recent checkpoint.

This will train the model according to the settings specified in the model config, using the number of training samples specified

To evaluate the model using metrics like CRPS, rank calculations, RMSE, RALSD, etc. on some of the checkpoints, pass the `--evaluate` flag. You must also specify if you want
to do this for all model checkpoints or just a selection. Do this using 

- `--eval-full`	  (all model checkpoints)
- `--eval-short`	  (recommended; the final 1/3rd of model checkpoints)
- `--eval-blitz`	  (the final 4 model checkpoints)
- `--eval-model-numbers N1 N2` (only evaluate the model numbers N1 and N2)


Note: these evaluations can take a long time!

As an example, if you wanted to train a model on 320000 samples, using data in DATA_FOLDER, and the model config in MODEL_CONFIG_PATH, and then evaluate the last four model checkpoints on 100 samples with 2 ensemble members each, you would run the following:

`python -m dsrnngan.main --eval-blitz --num-samples 320000 --restart --records-folder DATA_FOLDER --model-config-path MODEL_CONFIG_PATH --output-suffix my_model --num-eval-images 100 --eval-ensemble-size 2`

By default, the evaluation is performed over the date range specified in the model_config.yaml file. These can be overridden by specifiying start and end year-months via `--val-ym-start` and `--val-ym-start` in the form YYYYMM. 

If you would like the evaluation to be done without shuffling (e.g. to ensure consistency between model runs) then use the `--no-shuffle-eval` flag.

To save the samples that are generated during evaluation, for offline model analysis, use the  `--save-generated-samples` flag.

If you've already trained your model, and you just want to run some 
evaluation, use the `--no-train` flag, for example:

# Quantile mapping

In order to quantile map the cGAN, you will need to produce samples for the cGAN over the training year. Do this by running the main script in evaluate mode for the model number you have chosen, without shuffling, and save the generated samples E.g.:

`python -m dsrnngan.main --no-train --eval-model-numbers BEST_MODEL_NUM --save-generated-samples --model-folder TRAINED_MODEL_FOLDER --num-images 18000 --eval-ensemble-size 1 --eval-on-train-set`

In this case, the `--eval-on-train-set` flag means it will read the training data range from the config, but you can override this with `--val-ym-start` and `--val-ym-end` if needed.

Once this has finished, a folder will be created within TRAINED_MODEL_FOLDER, specific to the training range. Copy this path, and use it as an input to the quantile mapping script:

```bash
scripts.quantile_mapping [-h] --model-eval-folder MODEL_EVAL_FOLDER --model-number MODEL_NUMBER [--num-lat-lon-chunks NUM_LAT_LON_CHUNKS] [--output-folder OUTPUT_FOLDER] [--min-points-per-quantile MIN_POINTS_PER_QUANTILE] [--save-data]
               [--save-qmapper] [--plot] [--debug]

```
Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--model-eval-folder`|`None`|Folder containing evaluated samples for the model|
||`--model-number`|`None`|Checkpoint number of model that created the samples|
||`--num-lat-lon-chunks`|`1`|Number of chunks to split up spatial data into along each axis|
||`--output-folder`|`None`|Folder to save plots in|
||`--min-points-per-quantile`|`1`|Minimum number of data points per quantile in the plots|
||`--save-data`||Save the quantile mapped data to the model folder|
||`--save-qmapper`||Save the quantile mapping objects to the model folder|
||`--plot`||Make plots|
||`--debug`||Debug mode|

You may want to iterate over several different settings of `num_lat_lon_chunks`; one way of doing this is to run this script many times using `--plot` to create plots (and data) for different scenarios, then choosing the best one by eye / your metric of choice.


# Plotting

Use the script in `scripts/make_plots.py` to create plots and data for plots.

```
python -m scripts.make_plots --output-dir OUTPUT_DIR --nickname NICKNAME --model-eval-folder MODEL_EVAL_FOLDER --model-number MODEL_NUMBER [-ex] [-sc] [-rh] [-rmse] [-bias] [-se] [-rapsd] [-qq] [-hist] [-crps] [-fss] [-d] [-conf] [-csi] [--debug]
```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--output-dir`|`None`|Folder to store the plots in|
||`--nickname`|`None`|nickname to give this model|
||`--model-eval-folder`|`None`|Folder containing pre-evaluated cGAN data|
||`--model-number`|`None`|Checkpoint number of model|
|`-ex`|`--examples`||Plot a selection of example precipitation forecasts|
|`-sc`|`--scatter`||Plot scatter plots of domain averaged rainfall|
|`-rh`|`--rank-hist`||Plot rank histograms|
|`-rmse`|||Plot root mean square error|
|`-bias`|||Plot bias|
|`-se`|`--spread-error`||Plot the spread error|
|`-rapsd`|||Plot the radially averaged power spectral density|
|`-qq`|`--quantiles`||Create quantile-quantile plot|
|`-hist`|||Plot Histogram of rainfall intensities|
|`-crps`|||Plot CRPS scores|
|`-fss`|||PLot fractions skill score|
|`-d`|`--diurnal`||Plot diurnal cycle|
|`-conf`|`--confusion-matrix`||Calculate confusion matrices|
|`-csi`|||Plot CSI and ETS|
||`--debug`||Debug flag to use small amounts of data|

# Logging

Optional logging inside main.py used Weights and Biases; to use this functionality you need to create an account and follow their documentation to make sure the API calls will authenticate correctly. 
