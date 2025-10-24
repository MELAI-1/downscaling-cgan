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
#                 do_plot=False,
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

from dsrnngan.model import noise
from dsrnngan.evaluation import plots


def train_model(*,
                model=None,
                mode=None,
                batch_gen_train=None,
                # ... autres paramètres ...
                steps_per_checkpoint=None,
                training_ratio=5):

    # --- 1. ISOLER LA LECTURE DU PREMIER LOT POUR DÉTERMINER LES FORMES ---
    
    # Créer l'itérateur
    try:
        dataset_iterator = iter(batch_gen_train.as_numpy_iterator())
    except Exception as e:
        print(f"ERREUR CRITIQUE : Impossible de créer l'itérateur du générateur. Erreur : {e}")
        raise

    # Tenter de lire le premier lot (X, Y, Z)
    try:
        # L'unpacking est correct car vous avez confirmé 3 éléments.
        cond, _, _ = next(dataset_iterator)
        
        # Déterminer les formes
        img_shape = cond.shape[1:-1]
        batch_size = cond.shape[0]

        # Nettoyage
        del cond
        
    except StopIteration:
        print("ERREUR GRAVE : Le générateur de données d'entraînement (batch_gen_train) est VIDE.")
        # Lever une erreur pour forcer la boucle while à s'arrêter au lieu de se bloquer
        raise RuntimeError("Générateur d'entraînement vide, l'entraînement ne peut pas commencer.")
    
    except Exception as e:
        print(f"ERREUR LORS DU CHARGEMENT DU PREMIER LOT (Lecture directe) : {e}")
        raise e
    
    # --- 2. LE RESTE DU CODE RESTE INCHANGÉ ---

    if mode == 'GAN':
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        
        # L'appel de model.train utilisera le générateur DÉSORMAIS ÉPUISÉ d'un lot,
        # mais la méthode model.train est conçue pour relancer l'itération.
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_checkpoint, training_ratio=training_ratio)

    elif mode == 'VAEGAN':
        # ... (bloc VAEGAN inchangé) ...
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_checkpoint, training_ratio=5)

    elif mode == 'det':
        loss_log = model.train(batch_gen_train, steps_per_checkpoint)

    if do_plot:
        plots.plot_sequences(model.gen,
                             mode,
                             batch_gen_valid,
                             checkpoint,
                             noise_channels=noise_channels,
                             latent_variables=latent_variables,
                             num_samples=plot_samples,
                             out_fn=plot_fn)

    return loss_log
