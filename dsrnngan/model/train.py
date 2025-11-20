# import os
# import tensorflow as tf

# from dsrnngan.model import noise
# from dsrnngan.evaluation import plots


# def train_model(*,
#                 model=None,
#                 mode=None,
#                 batch_gen_train=None,
#                 batch_gen_valid=None,
#                 noise_channels=None,
#                 latent_variables=None,
#                 checkpoint=None,
#                 steps_per_checkpoint=None,
#                 do_plot=True,
#                 plot_samples=8,
#                 plot_fn=None,
#                 log_folder=None,
#                 training_ratio=5):

#     for cond, _, _ in batch_gen_train.take(1).as_numpy_iterator():
#         img_shape = cond.shape[1:-1]
#         batch_size = cond.shape[0]
#     del cond

#     if mode == 'GAN':
#         noise_shape = (img_shape[0], img_shape[1], noise_channels)
#         noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
#         loss_log = model.train(batch_gen_train, noise_gen,
#                                steps_per_checkpoint, training_ratio=training_ratio)

#     elif mode == 'VAEGAN':
#         noise_shape = (img_shape[0], img_shape[1], latent_variables)
#         noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
#         loss_log = model.train(batch_gen_train, noise_gen,
#                                steps_per_checkpoint, training_ratio=5)

#     elif mode == 'det':
#         loss_log = model.train(batch_gen_train, steps_per_checkpoint)

#     if do_plot:
#         print("Generating plots...")
#         print(f"Plotting to {os.path.abspath(plot_fn)}")
#         print(f"Using checkpoint {checkpoint}")
#         print(f"Using {plot_samples} samples")
#         print(f"Using mode {mode}")
#         print(f"Using noise channels {noise_channels}")
#         print(f"Using latent variables {latent_variables}")
#         print
#         plots.plot_sequences(model.gen,
#                             mode,
#                             batch_gen_valid,
#                             checkpoint,
#                             noise_channels=noise_channels,
#                             latent_variables=latent_variables,
#                             num_samples=plot_samples,
#                             out_fn=plot_fn)

#     return loss_log

import os
import tensorflow as tf
import numpy as np

from dsrnngan.model import noise
from dsrnngan.evaluation import plots


def inspect_batch_structure(batch, name="Data"):
    """Helper function to print data structure details."""
    print(f"\n--- INSPECTION: {name} ---")
    print(f"Type: {type(batch)}")
    
    # Cas 1 : Le batch est un tuple ou une liste (ex: (inputs, targets))
    if isinstance(batch, (tuple, list)):
        print(f"Length: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"  Item {i} Type: {type(item)}")
            if isinstance(item, dict):
                print(f"  Item {i} Keys: {item.keys()}")
                for k, v in item.items():
                    if hasattr(v, 'shape'):
                        print(f"    Key '{k}' shape: {v.shape}")
            elif hasattr(item, 'shape'):
                print(f"  Item {i} shape: {item.shape}")
                
    # Cas 2 : Le batch est directement un dictionnaire
    elif isinstance(batch, dict):
        print(f"Keys: {batch.keys()}")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  Key '{k}' shape: {v.shape}")
    
    print("--------------------------\n")


def train_model(*,
                model=None,
                mode=None,
                batch_gen_train=None,
                batch_gen_valid=None,
                noise_channels=None,
                latent_variables=None,
                checkpoint=None,
                steps_per_checkpoint=None,
                do_plot=True,
                plot_samples=8,
                plot_fn=None,
                log_folder=None,
                training_ratio=5):

    # -----------------------------------------------------------
    # DEBUG & INITIALIZATION BLOCK
    # -----------------------------------------------------------
    
    # 1. Inspect Train Data Structure
    try:
        # Tente de récupérer un batch (supporte tf.data.Dataset et générateurs Python)
        if hasattr(batch_gen_train, 'take'):
            iter_train = batch_gen_train.take(1).as_numpy_iterator()
        else:
            iter_train = iter(batch_gen_train)
            
        batch_train = next(iter_train)
        inspect_batch_structure(batch_train, name="batch_gen_train")
        
    except Exception as e:
        print(f"Error inspecting train batch: {e}")
        raise e

    # 2. Inspect Valid Data Structure (if exists)
    if batch_gen_valid is not None:
        try:
            if hasattr(batch_gen_valid, 'take'):
                iter_valid = batch_gen_valid.take(1).as_numpy_iterator()
            else:
                iter_valid = iter(batch_gen_valid)
            
            batch_valid = next(iter_valid)
            inspect_batch_structure(batch_valid, name="batch_gen_valid")
        except Exception as e:
            print(f"Warning: Could not inspect validation batch: {e}")

    # 3. Robustly determine img_shape and batch_size from batch_train
    # This handles both Dict inputs and Tuple inputs
    inputs = batch_train[0] if isinstance(batch_train, (list, tuple)) else batch_train
    
    if isinstance(inputs, dict):
        # Essayons de trouver la clé correspondant à l'image (souvent 'lo_res_inputs')
        if 'lo_res_inputs' in inputs:
            cond_tensor = inputs['lo_res_inputs']
        else:
            # Fallback: on prend la première valeur du dictionnaire
            print("Warning: 'lo_res_inputs' key not found. Using first value for shape inference.")
            cond_tensor = list(inputs.values())[0]
    else:
        # C'est probablement un tenseur direct (dans le cas d'une liste [cond, const, ...])
        cond_tensor = inputs

    # Calcul des dimensions
    # shape attendue: (batch_size, height, width, channels)
    img_shape = cond_tensor.shape[1:-1] 
    batch_size = cond_tensor.shape[0]
    
    print(f"Detected Image Shape: {img_shape}")
    print(f"Detected Batch Size: {batch_size}")
    print("-----------------------------------------------------------\n")

    # -----------------------------------------------------------
    # TRAINING LOGIC
    # -----------------------------------------------------------

    if mode == 'GAN':
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_checkpoint, training_ratio=training_ratio)

    elif mode == 'VAEGAN':
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_checkpoint, training_ratio=5)

    elif mode == 'det':
        loss_log = model.train(batch_gen_train, steps_per_checkpoint)

    if do_plot:
        print("Generating plots...")
        print(f"Plotting to {os.path.abspath(plot_fn)}")
        print(f"Using checkpoint {checkpoint}")
        print(f"Using {plot_samples} samples")
        print(f"Using mode {mode}")
        print(f"Using noise channels {noise_channels}")
        print(f"Using latent variables {latent_variables}")
        
        plots.plot_sequences(model.gen,
                            mode,
                            batch_gen_valid,
                            checkpoint,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            num_samples=plot_samples,
                            out_fn=plot_fn)

    return loss_log