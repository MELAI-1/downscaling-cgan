import argparse
import json
import os
import git
import wandb
import copy
import logging
from glob import glob
from types import SimpleNamespace
from tensorflow import config as tf_config

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

from dsrnngan.data import data, setupdata
from dsrnngan.evaluation import evaluation
from dsrnngan.utils import read_config, utils
from dsrnngan.model import setupmodel, train

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


parser = argparse.ArgumentParser()
parser.add_argument('--records-folder', type=str, default=None,
                    help="Folder from which to gather the tensorflow records")
model_config_group = parser.add_mutually_exclusive_group(required=False)

model_config_group.add_argument('--model-folder', type=str, default=None,
                    help="Folder in which previous model configs and training results have been stored.")
model_config_group.add_argument('--model-config-path', type=str, help='Full path of model config yaml file to use.')

parser.add_argument('--no-train', dest='do_training', action='store_false',
                    help="Do NOT carry out training, only perform eval")
parser.add_argument('--restart', dest='restart', action='store_true',
                    help="Restart training from latest checkpoint")

eval_group = parser.add_mutually_exclusive_group()
eval_group.add_argument('--eval-model-numbers', nargs='+', default=None,
                    help='Model number(s) to evaluate on (space separated)')
eval_group.add_argument('--eval-full', dest='evalnum', action='store_const', const="full", help="Evaluate on all checkpoints.")
eval_group.add_argument('--eval-short', dest='evalnum', action='store_const', const="short", help="Evaluate on last third of checkpoints")
eval_group.add_argument('--eval-blitz', dest='evalnum', action='store_const', const="blitz", help="Evaluate on last 4 checkpoints.")

parser.add_argument('--num-samples', type=int,
                    help="Number of samples to train on (overrides value in config)")
parser.add_argument('--num-eval-images', type=int, default=20,
                    help="Number of images to evaluate on")
parser.add_argument('--eval-ensemble-size', type=int, default=None,
                    help="Size of ensemble to evaluate on")
parser.add_argument('--eval-on-train-set', action="store_true", 
                    help="Use this flag to make the evaluation occur on the training dates")
parser.add_argument('--noise-factor', type=float, default=1e-3,
                    help="Multiplicative noise factor for rank histogram")
parser.add_argument('--val-ym-start', type=str,
                    help='Validation start in YYYYMM format (defaults to range specified in config)')
parser.add_argument('--val-ym-end', type=str,
                    help='Validation start in YYYYMM format (defaults to the range specified in the config)')
parser.add_argument('--no-shuffle-eval', action='store_true', 
                    help='Boolean, will turn off shuffling at evaluation.')
parser.add_argument('--save-generated-samples', action='store_true',
                    help='Flag to trigger saving of the evaluation arrays')
parser.add_argument('--training-weights', default=None, nargs=4,help='Weighting of classes to use in training (assumes data has been split according to e.g. mean rainfall)',
                    type=float
                    )
parser.add_argument('--output-suffix', default=None, type=str,
                    help='Suffix to append to model folder. If none then model folder has same name as TF records folder used as input.')
parser.add_argument('--log-folder', type=str, default=None, help='Root folder to which models are saved')
parser.add_argument('--wandb-logging', action='store_true', help="Flag to trigger logging to weights and biases")          
parser.add_argument('--debug', action='store_true', help="Flag to trigger debug mode, to reduce data volumes")                


def main(restart: bool, 
         do_training: bool, 
         num_eval_images: int,
         noise_factor: float, 
         eval_ensemble_size: int, 
         model_config: SimpleNamespace,
         data_config: SimpleNamespace,
         records_folder: str,
         data_paths: dict=None,
         shuffle_eval: bool=True, 
         evalnum: str=None, 
         eval_model_numbers: list=None, 
         seed: int=None, 
         num_samples_override: int=None,
         val_start: str=None, 
         val_end: str=None, 
         save_generated_samples: bool=False, 
         training_weights: list=None, 
         debug: bool=False,
         output_suffix: str=None, 
         log_folder: str=None,
         wandb_logging: bool=False
         ):
    """ Function for training and evaluating a cGAN, from a dataset of tf records

    Args:
        restart (bool): If True then will restart training from latest checkpoint       
        do_training (bool): If True then will run training on model 
        num_images (int): Number of images to evaluate on
        noise_factor (float): Noise factor to add to CRPS evaluation
        eval_ensemble_size (int): Size of ensemble to evaluate on
        shuffle_eval (bool, optional): If False then no shuffling of data before evaluation (for cases where you want to assess on a contiguous range of dates). Defaults to True.
        records_folder (str, optional): Location of tfrecords. Defaults to None. If None then will try to infer folder from tehash of the config.
        evalnum (str, optional): String representing type of evaluation: blitz, full, short. Defaults to None.
        eval_model_numbers (list, optional): List of models numbers to evaluate. Defaults to None.
        seed (int, optional): Optional random seed. Defaults to None.
        num_samples_override (int, optional): Total of samples to use in traininig. Defaults to None. If None then will read from config
        val_start (str, optional): Validation start in YYYYMM format. Defaults to None.
        val_end (str, optional): Validation end in YYYYMM format. Defaults to None.
        save_generated_samples (bool, optional): If True then generated samples are stored for further analysis. Defaults to False.
        training_weights (list, optional): List of floats, fraction of samples to take from each class in the training data. Defaults to None. If None then is read from the config
        debug (bool, optional): Whether or not to use in debug mode (stops training after first checkpoint). Defaults to False.
        output_suffix (str, optional): Suffix to append to output folder name. Defaults to None.
        log_folder (str, optional): root folder to store results in. Defaults to None. If None then will be taken from config.
    
    """

    
    # Create dicts for saving and hashing
    model_config_dict = utils.convert_namespace_to_dict(model_config)
    data_config_dict = utils.convert_namespace_to_dict(data_config)
                
    if not output_suffix:
        # Attach model_config as suffix
        output_suffix = '_' + utils.hash_dict(model_config_dict)
    else:
        output_suffix = '_' + output_suffix if not output_suffix.startswith('_') else output_suffix
        
    log_folder = log_folder + output_suffix
    print("Models being written to folder: ", log_folder, flush=True)      

    num_constant_fields = len(data_config.constant_fields)

    input_image_shape = (data_config.input_image_height, data_config.input_image_width, data_config.input_channels)
    output_image_shape = (model_config.downscaling_factor * input_image_shape[0], model_config.downscaling_factor * input_image_shape[1], 1)
    constants_image_shape = (data_config.input_image_height, data_config.input_image_width, num_constant_fields)
    
    if training_weights is None:
        training_weights = model_config.train.training_weights

    if (val_start is not None) and (val_end is not None):
        model_config.val.val_range = [[val_start, val_end]]
    elif val_start is not None or val_end is not None:
        raise ValueError('Must specify both val_start and val_end or neither.')

    
    noise_factor = float(noise_factor)
    evaluate = (eval_model_numbers or evalnum)
    
    if evaluate and model_config.val.val_range is None:
        raise ValueError('Must specify validation range when using --evaluate flag')
    
    # Calculate number of samples from epochs:
    if num_samples_override is not None:
        model_config.train.num_samples = num_samples_override
    else:
        if model_config.train.num_samples is None:
            num_data_points = len(utils.date_range_from_year_month_range(model_config.train.training_range)) * len(data.all_fcst_hours)
            model_config.train.num_samples = num_data_points * model_config.train.num_epochs
        
    num_checkpoints = int(model_config.train.num_samples/(model_config.train.steps_per_checkpoint * model_config.train.batch_size))
    checkpoint = 1
    
    # Get Git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # create root log folder / log folder / models subfolder if they don't exist
    # Path(root_log_folder).mkdir(parents=True, exist_ok=True)
    # log_folder = os.path.join(root_log_folder, sha[:8])
    
    model_weights_root = os.path.join(log_folder, "models")
    os.makedirs(model_weights_root, exist_ok=True)

    # save setup parameters
    utils.write_to_yaml(os.path.join(log_folder, 'data_config.yaml'), data_config_dict)
    utils.write_to_yaml(os.path.join(log_folder, 'model_config.yaml'), model_config_dict)
        
    with open(os.path.join(log_folder, 'git_commit.txt'), 'w+') as ofh:
        ofh.write(sha)

    if do_training:
        logger.debug('Starting training')
        # initialize GAN
        model = setupmodel.setup_model(
            model_config=model_config,
            data_config=data_config)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            data_config=data_config,
            model_config=model_config,
            records_folder=records_folder,
            fcst_shape=input_image_shape,
            con_shape=constants_image_shape,
            out_shape=output_image_shape,
            weights=training_weights,
            load_full_image=False,
            seed=seed)

        if restart: # load weights and run status
            try:
                model.load(model.filenames_from_root(model_weights_root))
                with open(os.path.join(log_folder, "run_status.json"), 'r') as f:
                    run_status = json.load(f)
                training_samples = run_status["training_samples"]
                checkpoint = int(training_samples / (model_config.train.steps_per_checkpoint * model_config.train.batch_size)) + 1

                log_file = "{}/log.txt".format(log_folder)
                log = pd.read_csv(log_file)
                log_list = [log]
                
                print(f'Loading model from previously trained checkpoint: {checkpoint}')
            except FileNotFoundError:
                # Catch case where folder exists but no model saved yet
                training_samples = 0

                log_file = os.path.join(log_folder, "log.txt")
                log_list = []

        else:  # initialize run status
            training_samples = 0

            log_file = os.path.join(log_folder, "log.txt")
            log_list = []

        # plot_fname = os.path.join(log_folder, "progress.pdf")
        plot_fname = os.path.join(log_folder, f"progress") 

        while (training_samples < model_config.train.num_samples):  # main training loop

            if not wandb_logging:
                mode = 'disabled'
            else:
                mode = 'online'
                # Initiialise weights and biases logging
            wandb.init(
                entity=model_config.wandb.entity,
                project=model_config.wandb.debug_project_name if args.debug else model_config.wandb.project_name,
                sync_tensorboard=True,
                name=log_folder.split('/')[-1],
                config={
                    'data': data_config_dict,
                    'model': model_config_dict,
                    'gpu_devices': tf_config.list_physical_devices('GPU')
                },
                mode=mode
            )
                

            logger.debug("Checkpoint {}/{}".format(checkpoint, num_checkpoints))

            # train for some number of batches
            loss_log = train.train_model(model=model,
                                         mode=model_config.mode,
                                         batch_gen_train=batch_gen_train,
                                         batch_gen_valid=batch_gen_valid,
                                         noise_channels=model_config.generator.noise_channels,
                                         latent_variables=model_config.generator.latent_variables,
                                         checkpoint=checkpoint,
                                         steps_per_checkpoint=model_config.train.steps_per_checkpoint,
                                         plot_samples=model_config.val.val_size,
                                         plot_fn=plot_fname,
                                         training_ratio=model_config.train.training_ratio)

            training_samples += model_config.train.steps_per_checkpoint * model_config.train.batch_size
            checkpoint += 1


            # save results
            model.save(model_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(os.path.join(log_folder, "run_status.json"), 'w') as f:
                json.dump(run_status, f)

            log_data = {"training_samples": [training_samples]}
            for foo in loss_log:
                log_data[foo] = loss_log[foo]

            log_list.append(pd.DataFrame(data=log_data))
            log = pd.concat(log_list)
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each checkpoint
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)
            
            if debug:
                break

    else:
        logger.debug("Training skipped...")
        
    # model iterations to save full rank data to disk for during evaluations;
    # necessary for plot rank histograms. these are large files, so small
    # selection used to avoid storing gigabytes of data
    interval = model_config.train.steps_per_checkpoint * model_config.train.batch_size
    
    # Get latest model number
    latest_model_fp = sorted(glob(os.path.join(model_weights_root, '*.h5')))[-1]
    finalchkpt = int(int(latest_model_fp.split('/')[-1].split('.')[-2][-7:]) / interval)
    
    # last 4 checkpoints, or all checkpoints if < 4
    ranks_to_save = [(finalchkpt - ii)*interval for ii in range(3, -1, -1)] if finalchkpt >= 4 else [ii*interval for ii in range(1, finalchkpt+1)]

    if eval_model_numbers:
        ranks_to_save = eval_model_numbers
        pass
    elif evalnum == "blitz":
        eval_model_numbers = ranks_to_save.copy()  # should not be modifying list in-place, but just in case!
    elif evalnum == "short":
        # last 1/3rd of checkpoints
        Neval = max(finalchkpt // 3, 1)
        eval_model_numbers = [(finalchkpt - ii)*interval for ii in range((Neval-1), -1, -1)]
    elif evalnum == "full":
        eval_model_numbers = np.arange(0, model_config.train.num_samples + 1, interval)[1:].tolist()

    # evaluate model performance
    if evaluate:
        logger.debug('Performing evaluation')
        evaluation.evaluate_multiple_checkpoints(model_config=model_config,
                                                 data_config=data_config,
                                                 log_folder=log_folder,
                                                 weights_dir=model_weights_root,
                                                 records_folder=records_folder,
                                                 noise_factor=noise_factor,
                                                 model_numbers=eval_model_numbers,
                                                 num_images=num_eval_images,
                                                 ensemble_size=eval_ensemble_size,
                                                 shuffle=shuffle_eval,
                                                 save_generated_samples=save_generated_samples,
                                                 batch_size=1,
                                                 use_training_data=args.eval_on_train_set)
    
    return log_folder


if __name__ == "__main__":
    
    print('Checking GPU devices')
    gpu_devices = tf_config.list_physical_devices('GPU')
    
    print(gpu_devices)
    
    if len(gpu_devices) == 0:
        logger.debug('GPU devices are not being seen')
    logger.debug(gpu_devices)
    
    print('Setting GPU mode')
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
    
    args = parser.parse_args()
    
    if args.eval_model_numbers:
        eval_model_numbers = [args.eval_model_numbers] if not isinstance(args.eval_model_numbers, list) else args.eval_model_numbers
        eval_model_numbers = [int(mn) for mn in eval_model_numbers]
    else:
        eval_model_numbers = None
    print(args, flush=True)
    
    if args.model_config_path is not None:
        path_split = os.path.split(args.model_config_path)
        model_config = read_config.read_model_config(config_filename=path_split[-1],
                                                     config_folder=path_split[0])
        data_config = None
        output_suffix = args.output_suffix
        
    elif args.model_folder is not None:
        if os.path.isfile(os.path.join(args.model_folder, 'setup_params.yaml')):
            # Retained for backwards compatability
            config_dict = read_config.read_config(config_filename='setup_params.yaml', config_folder=args.model_folder)
            model_config, data_config = read_config.get_config_objects(config=config_dict)
            
            data_config.paths = {'BLUE_PEBBLE': data.DATA_PATHS}
        else:
            model_config = read_config.read_model_config(config_folder=args.model_folder)
            data_config = read_config.read_data_config(config_folder=args.model_folder)
        
        if '_' in args.model_folder:
            log_folder = '_'.join(args.model_folder.split('_')[:-1])
            output_suffix = args.model_folder.split('_')[-1] # If model folder specified then output suffix is redundant
        else:
            log_folder = args.model_folder
            output_suffix = None
        
        if args.records_folder is None:
            data_paths = read_config.get_data_paths(data_config=data_config)
        else:
            data_paths = read_config.get_data_paths(config_folder=args.records_folder)
    else:
        model_config = read_config.read_model_config()
        data_config = None
        output_suffix = args.output_suffix
    
    
    if data_config is None:
        if args.records_folder is None:
            
            data_config = read_config.read_data_config()
            data_paths = read_config.get_data_paths()
            
            records_folder = os.path.join(data_paths['TFRecords']['tfrecords_path'], utils.hash_dict(data_config.__dict__))
            if not os.path.isdir(records_folder):
                raise ValueError('Data has not been prepared that matches this data config')
        else:
            data_config = read_config.read_data_config(config_folder=args.records_folder)
            data_paths = read_config.get_data_paths(config_folder=args.records_folder)
    
        log_folder = args.log_folder or model_config.log_folder 
        log_folder = os.path.join(log_folder, args.records_folder.split('/')[-1])

    main(
            model_config=model_config,
            data_config=data_config,
            data_paths=data_paths,
            records_folder=args.records_folder,
            restart=args.restart, 
            do_training=args.do_training, 
            evalnum=args.evalnum,
            noise_factor=args.noise_factor,
            num_samples_override=args.num_samples,
            num_eval_images=args.num_eval_images,
            eval_model_numbers=eval_model_numbers,
            val_start=args.val_ym_start,
            val_end=args.val_ym_end,
            eval_ensemble_size=args.eval_ensemble_size,
            shuffle_eval=not args.no_shuffle_eval,
            save_generated_samples=args.save_generated_samples,
            training_weights=args.training_weights,
            output_suffix=output_suffix,
            log_folder=log_folder,
            debug=args.debug,
            wandb_logging=args.wandb_logging)
