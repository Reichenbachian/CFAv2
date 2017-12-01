#### System
import sys
import os

### DlFramework
from dlframework.filters.general_filters import array_length_ok
from dlframework.augmentations.general_augmentations import identity, add_noise, \
                                                      remove_outliers, amplitude_extractor, \
                                                      subtract_gaussian_filter, offset, \
                                                      add_linear_trend
from dlframework.processors.general_processors import smooth_guassian_processor
from dlframework.processors.audio_processors import wav_normalize
from dlframework.keras_extensions.callbacks import ConfusionMatrix, ROCAnalysis
from dlframework.callbacks.classification_callbacks import LcGraph, precision_per_class, recall_per_class, pred_per_class, LabelHistogram

### Data Manipulation
import pandas as pd
import keras

### DL
from functools import partial
from keras.callbacks import Callback
from keras import backend as K
from keras.optimizers import SGD
import tensorflow as tf
from keras.callbacks import EarlyStopping
from functools import partial

class planet_configuration(object):
    #### Names
    experiment_name = 'lstm_exp_1'
    local_cache_directory = 'local_cache/'


    #### Input
    input_shape = (1360, 1)

    #### Output
    mapped_label_list = ["planet", "other"]
    num_classes = len(mapped_label_list)
    saving_path = 'LRCN/' + experiment_name
    model_path = 'models/PlanetDiscriminator.json'
    weights_path= 'models/PlanetDiscriminator.h5'
    
    #### Controllers
    filters     = [partial(array_length_ok, min_length=1360)]
    augmentors = [identity,
                  identity,
                  # partial(add_noise, scale=.5),
                  # partial(add_noise, scale=.5),
                  # partial(offset, max_offset_scale=3),
                  # partial(add_linear_trend, scale=3),
                  # add_noise,
                  # add_noise
                  ]
    processors = [# partial(subtract_gaussian_filter, k=5),
                  # remove_outliers,
                  # amplitude_extractor,
                  # controllers.remove_trend_poly,
                  # smooth_guassian_processor,
                  # split_periods,
                  # sum_along_period
                  ]
    
    #### Training
    train_steps_per_epoch = 1000
    nb_epochs = 100000
    save_every = 100
    validation_steps = 91
    num_epochs = 100000
    less_epochs = 100
    save_every = 1
    
    model_compilation_args = {
                               'optimizer': keras.optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8),
                               'loss': ['categorical_crossentropy'],
                               'metrics': ['accuracy']
                             }

    #### Callbacks
    tensorboard_dir = os.path.join(saving_path,'results/')
    
    callbacks = [keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_images= True),
                LcGraph(tensorboard_dir, mapped_label_list, interval=1),
                LabelHistogram(tensorboard_dir, mapped_label_list, interval=2),
                ConfusionMatrix(log_dir=tensorboard_dir, interval=2),
                ROCAnalysis(log_dir=tensorboard_dir, interval=2),
                EarlyStopping()]

    # Snapshot Callback is a default callback and needs the user to give config parameters at all times
    snapshot_parameters = {'filepath': os.path.join(saving_path,'weights',
                                                    experiment_name+'_weights.{epoch:02d}-{val_acc:.2f}.hdf5'),
                           'save_best': True,       #optional
                           'filepath_best': os.path.join(saving_path,'weights',
                                                         experiment_name+'_weights_best_model.{epoch:02d}-{val_acc:.2f}.hdf5'),
                           'period': 5,              #optional
                           'monitor':'val_acc',     #optional
                           'mode':'max'
                          }
    
    #### Per class Callbacks
    # Add per class precision and recall.
    custom_metrics = []
    # Recall
    for i, lbl_name in enumerate(mapped_label_list):
        p = partial(recall_per_class, cls=i)
        p.__name__ = "recall_"+lbl_name
        custom_metrics.append(p)
    # Precision
    for i, lbl_name in enumerate(mapped_label_list):
        p = partial(precision_per_class, cls=i)
        p.__name__ = "precision_"+lbl_name
        custom_metrics.append(p)
    # # Counted predicted
    # for i, lbl_name in enumerate(mapped_label_list):
    #     p = partial(pred_per_class, cls=i)
    #     p.__name__ = "predicted_"+lbl_name
    #     custom_metrics.append(p)

    model_compilation_args['metrics'] += custom_metrics


config = planet_configuration
