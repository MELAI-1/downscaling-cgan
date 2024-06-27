import gc
import os
import warnings
from datetime import datetime, timedelta
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from properscoring import crps_ensemble
from tensorflow.python.keras.utils import generic_utils
from datetime import datetime
from typing import Iterable
from types import SimpleNamespace

from dsrnngan.data.data_generator import DataGenerator
from dsrnngan.data import data
from dsrnngan.utils import read_config, utils
from dsrnngan.data import setupdata
from dsrnngan.model import setupmodel
from dsrnngan.model.noise import NoiseGenerator
from dsrnngan.model.pooling import pool
from dsrnngan.evaluation.rapsd import rapsd
from dsrnngan.evaluation.scoring import rmse, mse, mae, calculate_pearsonr, fss, critical_success_index
from dsrnngan.model import gan

warnings.filterwarnings("ignore", category=RuntimeWarning)


path = os.path.dirname(os.path.abspath(__file__))
model_config = read_config.read_model_config()
ds_fac = model_config.downscaling_factor

metrics = {'correlation': calculate_pearsonr, 'mae': mae, 'mse': mse,
           }

def setup_inputs(
                 model_config: SimpleNamespace,
                 data_config: SimpleNamespace,
                 records_folder: str,
                 hour: int,
                 shuffle: bool,
                 batch_size: int,
                 use_training_data: bool=False):

    # initialise model
    model = setupmodel.setup_model(model_config=model_config,
                                   data_config=data_config)

    gen = model.gen
    

    # always uses full-sized images
    print('Loading full sized image dataset')
    data_gen_train, data_gen_valid = setupdata.setup_data(
        data_config,
        model_config,
        records_folder=records_folder,
        load_full_image=True,
        shuffle=shuffle,
        hour=hour,
        full_image_batch_size=batch_size)
    
    if use_training_data:
        return gen, data_gen_train
    else:
        return gen, data_gen_valid


def _init_VAEGAN(gen, data_gen, load_full_image, batch_size, latent_variables):
    if False:
        # this runs the model on one batch, which is what the internet says
        # but this doesn't actually seem to be necessary?!
        data_gen_iter = iter(data_gen)
        if load_full_image:
            inputs, outputs = next(data_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
        else:
            cond, const, _ = next(data_gen_iter)

        noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        mean, logvar = gen.encoder([cond, const])
        gen.decoder.predict([mean, logvar, noise_gen(), const])
    # even after running the model on one batch, this needs to be set(?!)
    gen.built = True
    return

def generate_gan_sample(gen: gan.WGANGP, 
                        cond: np.ndarray, 
                        const: np.ndarray, 
                        noise_channels: int, 
                        ensemble_size: int=1, 
                        batch_size: int=1, 
                        seed: int=None):
    
    samples_gen = []
    noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
    noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size, random_seed=seed)
    for ii in range(ensemble_size):
        nn = noise_gen()
        sample_gen = gen.predict([cond, const, nn])
        samples_gen.append(sample_gen.astype("float32"))
        
    return samples_gen

def create_single_sample(*,
                   data_config: SimpleNamespace,
                   model_config: SimpleNamespace,
                   data_idx: int,
                   batch_size: int,
                   gen: gan.WGANGP,
                   data_gen: DataGenerator,
                   latitude_range: Iterable,
                   longitude_range: Iterable,
                   ensemble_size: int,
                   output_normalisation: str,
                   input_normalisation: str,
                   seed: int=None
                   ):
    
    if 'tp' in data_config.input_fields:
        tpidx = data_config.input_fields.index('tp')
    else:
        # Temporary fix to allow testing of other tp variants
        tpidx = data_config.input_fields.index('tpq')
    
    batch_size = 1  # do one full-size image at a time

    if model_config.mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    # load truth images
    inputs, outputs = data_gen[data_idx]
      
    cond = inputs['lo_res_inputs']
    fcst = cond[0, :, :, tpidx]
    const = inputs['hi_res_inputs']
    obs = outputs['output'][0, :, :]
    date = inputs['dates']
    hour = inputs['hours']           
    
    # Get observations 24hrs before
    imerg_persisted_fcst = data.load_imerg(date[0] - timedelta(days=1), hour=hour[0], latitude_vals=latitude_range, 
                                            longitude_vals=longitude_range, normalisation_type=output_normalisation)
    
    assert imerg_persisted_fcst.shape == obs.shape, ValueError('Shape mismatch in iMERG persistent and truth')
    assert len(date) == 1, ValueError('Currently must be run with a batch size of 1')
    assert len(date) == len(hour), ValueError('This is strange, why are they different sizes?')
        
    if output_normalisation is not None:
        obs = data.denormalise(obs, normalisation_type=output_normalisation)
    
    if input_normalisation is not None:
        fcst = data.denormalise(fcst, normalisation_type=input_normalisation)
            
    if model_config.mode == "GAN":
        
        samples_gen = generate_gan_sample(gen, cond, const, model_config.generator.noise_channels, ensemble_size, batch_size, seed)
            
    elif model_config.mode == "det":
        samples_gen = []
        sample_gen = gen.predict([cond, const])
        samples_gen.append(sample_gen.astype("float32"))
        
    elif model_config.mode == 'VAEGAN':
        samples_gen = []
        # call encoder once
        mean, logvar = gen.encoder([cond, const])
        noise_shape = np.array(cond)[0, ..., 0].shape + (model_config.latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        
        for ii in range(ensemble_size):
            nn = noise_gen()
            # generate ensemble of preds with decoder
            sample_gen = gen.decoder.predict([mean, logvar, nn, const])
            samples_gen.append(sample_gen.astype("float32"))

    # samples generated, now process them (e.g., undo log transform) and calculate MAE etc
    for ii in range(ensemble_size):
        
        sample_gen = samples_gen[ii][0, :, :, 0]
        
        # sample_gen shape should be [n, h, w, c] e.g. [1, 940, 940, 1]
        if output_normalisation is not None:
            sample_gen = data.denormalise(sample_gen, normalisation_type=output_normalisation)

        samples_gen[ii] = sample_gen

    return obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour

def eval_one_chkpt(*,
                   gen,
                   data_config,
                   model_config,
                   data_gen,
                   num_images,
                   noise_factor,
                   ensemble_size,
                   normalize_ranks=True,
                   show_progress=True,
                   batch_size: int=1):
    
    latitude_range, longitude_range = read_config.get_lat_lon_range_from_config(data_config)
    output_normalisation = data_config.output_normalisation
    
    if data_config.normalise_inputs:
        input_normalisation = data_config.input_normalisation_strategy['tp']['normalisation']
    else:
        input_normalisation = None
        
    if num_images < 5:
        Warning('These scores are best performed with more images')
    
    truth_vals = []
    samples_gen_vals = []
    fcst_vals = []
    persisted_fcst_vals = []
    dates = []
    hours = []
    cond_vals = []
    const_vals = []
    
    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    mse_all = []
    emmse_all = []
    fcst_mse_all = []
    ralsd_rmse_all = []
    ralsd_rmse_fcst_all = []
    corr_all = []
    correlation_fcst_all = []
    csi_fcst_all = []
    csi_all = []
    max_bias_all = []
    max_bias_fcst_all = []
    fcst_mae_all = []
    max_quantile_diff = []

    if 'tp' in data_config.input_fields:
        tpidx = data_config.input_fields.index('tp')
    else:
        # Temporary fix to allow testing of other tp variants
        tpidx = data_config.input_fields.index('tpq')
    
    batch_size = 1  # do one full-size image at a time

    if model_config.mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    if show_progress:
        # Initialize progbar
        progbar = generic_utils.Progbar(num_images,
                                        stateful_metrics=("CRPS", "EM-MSE"))

    CRPS_pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']

    rng = np.random.default_rng()

    data_idx = 0
    for kk in tqdm(range(num_images)):
        
        success = False
        for n in range(5):
            if success:
                continue
            try:
                obs, samples_gen, fcst, imerg_persisted_fcst, cond, const, date, hour = create_single_sample(
                        data_idx=data_idx,
                        data_config=data_config,
                        model_config=model_config,
                        batch_size=batch_size,
                        gen=gen,
                        data_gen=data_gen,
                        latitude_range=latitude_range,
                        longitude_range=longitude_range,
                        ensemble_size=ensemble_size,
                        output_normalisation=output_normalisation,
                        input_normalisation=input_normalisation
                        )
                success = True
                data_idx += 1
                
            except Exception as e:
                if n < 4:
                    print(f'Could not load file, attempting retry {n+1} of 4')
                    data_idx += 1
                    continue
                else:
                    raise e
            except IndexError:
                # Run out of samples
                break 
        
        dates.append(date)
        hours.append(hour)
        cond_vals.append(cond)
        const_vals.append(const)

        # Do all RALSD at once, to avoid re-calculating power spectrum of truth image
    
        ralsd_rmse = calculate_ralsd_rmse(obs, samples_gen)
        ralsd_rmse_all.append(ralsd_rmse.flatten())
        
        ralsd_rmse_fcst_all.append(calculate_ralsd_rmse(obs, [fcst]).flatten())
               
        # turn list of predictions into array, for CRPS/rank calculations
        samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]
        
        mse_all.append(mse(samples_gen[:,:,0], obs))
        emmse_all.append(mse(np.mean(samples_gen, axis=-1), obs))
        fcst_mse_all.append(mse(obs, fcst))
        fcst_mae_all.append(mae(obs,fcst))
        
        corr_all.append(calculate_pearsonr(obs, samples_gen[:, :, 0]))
        correlation_fcst_all.append(calculate_pearsonr(obs, fcst))
        
        # critical success index
        csi_all.append(critical_success_index(obs, samples_gen[:,:,0], threshold=1.0))
        csi_fcst_all.append(critical_success_index(obs, fcst, threshold=1.0))
        
        # Max difference
        max_bias_all.append((samples_gen - np.stack([obs]*ensemble_size, axis=-1)).max())
        max_bias_fcst_all.append((fcst -obs).max())
        
        # Max difference at 99.99th quantile
        max_quantile_diff.append( np.abs(np.quantile(samples_gen[:,:,0], 0.99999) - np.quantile(obs, 0.99999)).max() )
        
        # Store these values for e.g. correlation on the grid
        truth_vals.append(obs)
        samples_gen_vals.append(samples_gen)
        fcst_vals.append(fcst)
        persisted_fcst_vals.append(imerg_persisted_fcst)
        
        ####################  CRPS calculation ##########################
        # calculate CRPS scores for different pooling methods
        for method in CRPS_pooling_methods:

            if method == 'no_pooling':
                truth_pooled = obs
                samples_gen_pooled = samples_gen
            else:
                truth_pooled = pool(obs, method)
                samples_gen_pooled = pool(samples_gen, method)
                
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_truth_input = np.expand_dims(obs, 0)
            crps_gen_input = np.expand_dims(samples_gen, 0)
            crps_score_grid = crps_ensemble(crps_truth_input, crps_gen_input)
            crps_score = crps_score_grid.mean()
            del truth_pooled, samples_gen_pooled
            gc.collect()

            if method not in crps_scores:
                crps_scores[method] = []

            crps_scores[method].append(crps_score)
        
        
        # calculate ranks; only calculated without pooling

        # Add noise to truth and generated samples to make 0-handling fairer
        # NOTE truth and sample_gen are 'polluted' after this, hence we do this last
        obs += rng.random(size=obs.shape, dtype=np.float32)*noise_factor
        samples_gen += rng.random(size=samples_gen.shape, dtype=np.float32)*noise_factor

        truth_flat = obs.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
        samples_gen_ranks = samples_gen.reshape((-1, ensemble_size))  # unknown batch size/img dims, known number of samples
        rank = np.count_nonzero(truth_flat[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
        ranks.append(rank)
        
        # keep track of input and truth rainfall values, to facilitate further ranks processing
        cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx], normalisation_type=input_normalisation).astype(np.float32), ds_fac, axis=-1), ds_fac, axis=-2)
        lowress.append(cond_exp.ravel())
        hiress.append(obs.astype(np.float32).ravel())
        del samples_gen_ranks, truth_flat
        gc.collect()

        if show_progress:
            emmse_so_far = np.sqrt(np.mean(emmse_all))
            crps_mean = np.mean(crps_scores['no_pooling'])
            losses = [("EM-MSE", emmse_so_far), ("CRPS", crps_mean)]
            progbar.add(1, values=losses)
            

    truth_array = np.stack(truth_vals, axis=0)
    samples_gen_array = np.stack(samples_gen_vals, axis=0)
    fcst_array = np.stack(fcst_vals, axis=0)
    persisted_fcst_array = np.stack(persisted_fcst_vals, axis=0)
    
    point_metrics = {}
        
    ralsd_rmse_all = np.concatenate(ralsd_rmse_all)
    
    for method in crps_scores:
        point_metrics['CRPS_' + method] = np.asarray(crps_scores[method]).mean()
     
    point_metrics['rmse'] = np.sqrt(np.mean(mse_all))
    point_metrics['emmse'] = np.sqrt(np.mean(emmse_all))
    point_metrics['mse_fcst'] = np.sqrt(np.mean(fcst_mse_all))
    point_metrics['mae_fcst'] = np.sqrt(np.mean(fcst_mse_all))
    point_metrics['ralsd'] = np.nanmean(ralsd_rmse_all)
    point_metrics['ralsd_fcst'] = np.nanmean(ralsd_rmse_fcst_all)
    point_metrics['corr'] = np.mean(corr_all)
    point_metrics['corr_fcst'] = np.mean(correlation_fcst_all)
    point_metrics['csi'] = np.mean(csi_all)
    point_metrics['csi_fcst'] = np.mean(csi_fcst_all)
    point_metrics['max_bias'] = np.mean(max_bias_all)
    point_metrics['max_bias_fcst'] = np.mean(max_bias_fcst_all)
    point_metrics['max_quantile_diff'] = np.mean(max_quantile_diff)
    
    ranks = np.concatenate(ranks)
    lowress = np.concatenate(lowress)
    hiress = np.concatenate(hiress)
    gc.collect()
    if normalize_ranks:
        ranks = (ranks / ensemble_size).astype(np.float32)
        gc.collect()
    rank_arrays = (ranks, lowress, hiress)
    
    arrays = {'truth': truth_array, 'samples_gen': samples_gen_array, 'fcst_array': fcst_array, 
              'persisted_fcst': persisted_fcst_array, 'dates': dates, 'hours': hours}

    return rank_arrays, point_metrics, arrays


def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def log_line(log_fname, line):
    with open(log_fname, 'a') as f:
        print(line, file=f)


def evaluate_multiple_checkpoints(
                                  model_config,
                                  data_config,
                                  weights_dir,
                                  records_folder,
                                  model_numbers,
                                  log_folder,
                                  noise_factor,
                                  num_images,
                                  ensemble_size,
                                  shuffle,
                                  save_generated_samples=False,
                                  batch_size: int=1,
                                  use_training_data: bool=False
                                  ):

    gen, data_gen = setup_inputs(model_config=model_config,
                                       data_config=data_config,
                                       records_folder=records_folder,
                                       hour='random',
                                       shuffle=shuffle,
                                       batch_size=batch_size,
                                       use_training_data=use_training_data)
    header = True

    for model_number in model_numbers:
        gen_weights_file = os.path.join(weights_dir, f"gen_weights-{model_number:07d}.h5")

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        if model_config.mode == "VAEGAN":
            _init_VAEGAN(gen, data_gen, True, 1, model_config.latent_variables)
        gen.load_weights(gen_weights_file)
        
        rank_arrays, agg_metrics, arrays = eval_one_chkpt(
                                             gen=gen,
                                             data_gen=data_gen,
                                             data_config=data_config,
                                             model_config=model_config,
                                             num_images=num_images,
                                             ensemble_size=ensemble_size,
                                             noise_factor=noise_factor,
                                             batch_size=batch_size)
        ranks, lowress, hiress = rank_arrays
        OP = rank_OP(ranks)
        

        if use_training_data:
            ym_range = model_config.train.training_range
        else:
            ym_range = model_config.val.val_range
            
        # save one directory up from model weights, in same dir as logfile
        # Assuming that it is very unlikely to have the same start and end of the range and have a collision on this hash
        range_hash = hashlib.sha256(str(ym_range).encode('utf-8')).hexdigest()[:5]
        output_folder = os.path.join(log_folder, f"n{num_images}_{ym_range[0][0]}-{ym_range[-1][-1]}_{range_hash}_e{ensemble_size}")
        
        os.makedirs(output_folder, exist_ok=True)

        # Save exact validation range
        with open(os.path.join(output_folder, 'val_range.pkl'), 'wb+') as ofh:
            pickle.dump(ym_range, ofh)
        
        # Create a dataframe of all the data (to ensure scores are recorded in the right column)
        df = pd.DataFrame.from_dict(dict(N=model_number, op=OP, **{k: [v] for k, v in agg_metrics.items()}))
        eval_fname = os.path.join(output_folder, f"eval_validation.csv")
        df.to_csv(eval_fname, header=header, mode='a', float_format='%.6f', index=False)
        header = False

        if save_generated_samples:
            with open(os.path.join(output_folder, f"arrays-{model_number}.pkl"), 'wb+') as ofh:
                pickle.dump(arrays, ofh, pickle.HIGHEST_PROTOCOL)
        

def calculate_ralsd_rmse(truth, samples):
    # check 'batch size' is 1; can rewrite to handle batch size > 1 if necessary
    
    truth = np.copy(truth)
    
    if len(truth.shape) == 2:
        truth = np.expand_dims(truth, (0))
    
    expanded_samples = []
    for sample in samples:
        if len(sample.shape) == 2:
            sample = np.expand_dims(sample, (0))
        expanded_samples.append(sample)
    samples = expanded_samples
    
    assert truth.shape[0] == 1, 'Incorrect shape for truth'
    assert samples[0].shape[0] == 1

    # truth has shape 1 x W x H
    # samples is a list, each of shape 1 x W x H

    # avoid producing infinite or misleading values by not doing RALSD calc
    # for images that are mostly zeroes
    if truth.mean() < 0.002:
        return np.array([np.nan])
    
    # calculate RAPSD of truth once, not repeatedly!
    fft_freq_truth = rapsd(np.squeeze(truth, axis=0), fft_method=np.fft)
    dBtruth = 10 * np.log10(fft_freq_truth)

    ralsd_all = []
    for pred in samples:
        if pred.mean() < 0.002:
            ralsd_all.append(np.nan)
        else:
            fft_freq_pred = rapsd(np.squeeze(pred, axis=0), fft_method=np.fft)
            dBpred = 10 * np.log10(fft_freq_pred)
            rmse = np.sqrt(np.nanmean((dBtruth-dBpred)**2))
            ralsd_all.append(rmse)
    return np.array(ralsd_all)


def get_diurnal_cycle(array: np.ndarray, 
                      dates: list, 
                      hours: list, 
                      longitude_range: Iterable, 
                      latitude_range: Iterable):
    
    hourly_sums = {}
    hourly_counts = {}

    n_samples = array.shape[0]

    for n in range(n_samples):
        
        h = hours[n]
        d = dates[n]

        for l_ix, long in enumerate(longitude_range):
            
            local_hour = utils.get_local_datetime(utc_datetime=datetime(d.year, d.month, d.day, h),
                                                  longitude=long, latitude=np.mean(latitude_range)).hour

            if local_hour not in hourly_counts:
                hourly_counts[local_hour] = 1
                hourly_sums[local_hour] = array[n,:,l_ix].mean()
            else:
                hourly_counts[local_hour] += 1
                hourly_sums[local_hour] += array[n,:,l_ix].mean()
    return hourly_sums, hourly_counts

def get_fss_scores(truth_array, fss_data_dict, hourly_thresholds, window_sizes, n_samples):

    fss_results = {'thresholds': hourly_thresholds, 'window_sizes': window_sizes, 'scores': {}}

    for data_name, data_array in tqdm(fss_data_dict.items()):
        
        fss_results['scores'][data_name] = []

        for thr in hourly_thresholds:

            tmp_fss = []

            for w in tqdm(window_sizes):
                
                tmp_fss.append(fss(truth_array[:n_samples, :, :], data_array, w, thr, mode='constant'))
            
            fss_results['scores'][data_name].append(tmp_fss)
            
    return fss_results
