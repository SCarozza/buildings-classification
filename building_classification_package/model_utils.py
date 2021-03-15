from typing import Union

import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import DirectoryIterator as kerasDirectoryIterator

from building_classification_package.config import SYSTEM_CONFIG, \
    DATA_CONFIG, \
    load_model


def train_model(train_data: kerasDirectoryIterator, val_data: kerasDirectoryIterator, 
                model_config: dict, data_config: dict = DATA_CONFIG, system_config: dict = SYSTEM_CONFIG)\
                -> Union[tf.keras.Model, keras.Model]:
    """
    Train the model. Which model to train, which loss, optimizer to use and callbacks are defined in model_config.
    The model weights are saved in a file

    :param train_data: iterator of training data, divided in one folder for each class
    :param val_data: iterator of validation data, divided in one folder for each class. None if we don't want to
    validate during training (but for example use evaluate_model function later)
    :param model_config: dict of configuration for the model, including model object, optimizer, loss, metrics,
    callbacks, path to file where to save weights.
    :param data_config
    :param system_config
    :return: trained model
    """

    model = load_model(model_config, data_config, system_config)

    model.compile(optimizer=model_config['train']['optimizer'],
                  loss=model_config['train']['loss'],
                  metrics=model_config['train']['metrics'])

    if val_data is None:  # I add this condition because if I just pass a None to val_data the kernel crashes (?)
        model.fit(train_data, 
                  epochs=model_config['train']['epochs'],
                  callbacks=model_config['train']['callbacks'],
                  use_multiprocessing=False)
    else:
        model.fit(train_data,
                  validation_data=val_data,
                  epochs=model_config['train']['epochs'],
                  callbacks=model_config['train']['callbacks'],
                  use_multiprocessing=False)

    # unfortunately I cannot get to work the simple save functions
    # of keras and tf.keras. So I will save weights and reload the model instead
    model.save_weights(model_config['saved_weight_path'])

    return model


def load_trained_and_compiled_model(model_config: dict, data_config: dict = DATA_CONFIG,
                                    system_config: dict = SYSTEM_CONFIG) -> Union[tf.keras.Model, keras.Model]:
    """
    Load the architecture and weights of a trained model. Needed to do this step because the direct load_model functions
    from keras didn't work for me

    :param model_config: dict with model configurations
    :param data_config: dict with data configurations
    :param system_config: dict with system configurations
    :return: trained model object
    """
    # loaded_model = model_config['model']
    loaded_model = load_model(model_config, data_config, system_config)
    loaded_model.compile(optimizer=model_config['train']['optimizer'],
                         loss=model_config['train']['loss'],
                         metrics=model_config['train']['metrics'])
    loaded_model.load_weights(model_config['saved_weight_path'])
    return loaded_model


def evaluate_model(val_data: kerasDirectoryIterator, model_config: dict) -> np.ndarray:
    """
    Evaluate the model on a chosen dataset.

    :param val_data: iterator of data to use to evaluate the model
    :param model_config: dict of model configuration
    :return: array containing loss and accuracy of the model
    """
    model = load_trained_and_compiled_model(model_config)
    return model.evaluate(val_data)


def model_predict(model_config: dict, data: kerasDirectoryIterator) -> np.ndarray:
    """Predict data based on the model stored in model config

    :param model_config: dict of model configuration
    :param data: data to generate predictions of
    :return: array of prediction"""
    model = load_trained_and_compiled_model(model_config)
    return model.predict(data)
