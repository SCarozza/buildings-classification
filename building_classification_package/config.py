import pickle
from typing import Union

import keras
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD

from building_classification_package.callbacks import PlotLearning


# the system config is needed because if I am on dap I cannot access internet, so I cannot download the pre-trained
# vgg model. in this case, I should have a pickle saved in the location specified
SYSTEM_CONFIG = {
    'on_dap': False,
    'model_pickle_path': \
      '/opt/app-root/projects/Building_classification/buildings_classification/model_pickles/base_model_vgg256.pickle'
}

#in data config deecide how many classes we want to use of the 8 available in this data.
DATA_CONFIG = {'classes': ['apartment',
                            'house',
                            'industrial',
                            'retail',
                            'officebuilding',
                    #           'garage',
                    #           'church',
                    #           'roof'
                            ],
               'resize_img': False,
               'preprocessing_fuction': preprocess_input_vgg}


class SimpleCnnModel(tf.keras.Model):
    """
    I instanciate a simple cnn model as a submodel of the tf keras class. see:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model
    to make it work I have to make sure the layers I am using are of the tf.keras.layers kind, not keras API kind.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(SimpleCnnModel, self).__init__()

        self.conv1 = tfkl.Conv2D(16, (3, 3), name='conv1')
        self.batchnorm1 = tfkl.BatchNormalization(axis=3, name='bn1')
        self.activation1 = tfkl.Activation('relu')
        self.maxpool1 = tfkl.MaxPooling2D(2, name='max_pool1')

        self.conv2 = tfkl.Conv2D(25, (6, 6), name='conv2')
        self.batchnorm2 = tfkl.BatchNormalization(axis=3, name='bn2')
        self.activation2 = tfkl.Activation('relu')
        self.maxpool2 = tfkl.MaxPooling2D(2, name='max_pool2')
        self.dropout1 = tfkl.Dropout(0.5, name='drop1') 

        self.flatten = tfkl.Flatten(name='flatten')
        self.dense1 = tfkl.Dense(150, activation='relu', name='dense1')
        self.dropout2 = tfkl.Dropout(0.5, name='drop2')
        self.dense2 = tfkl.Dense(n_classes, activation='softmax', name='final_dense')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


def extend_pretrained_model(base_model: keras.Model, data_config: dict, n_freezed_layers: int = None) -> keras.Model:
    """
    Function to create a model with a pre-trained base model (e.g. VGG16). Here I could not use the tf.keras Model
    subclass method because it doesn't work with pre-trained net.

    :param base_model: pre-trained keras model instance. e.g. VGG16
    :param data_config: dict containing data configuration
    :param n_freezed_layers: number of layers to freeze. if None, by default if freezes all the base model layers
    :return: a keras model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.3)(x)
    #     x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(len(data_config['classes']), activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=predictions)

    if not n_freezed_layers:
        n_freezed_layers = len(base_model.layers)
    for layer in model.layers[:n_freezed_layers]:
        layer.trainable = False
    for layer in model.layers[n_freezed_layers:]:
        layer.trainable = True

    return model


def load_base_model(system_config: dict = SYSTEM_CONFIG) -> keras.Model:
    """
    This function calls the base model. I had to make this function because the base model is obtained differently
    if I am on DAP or not. DAP doesn't have access to internet so we cannot download the weights of VGG16, so a
    pickle must be saved beforehand.

    :param system_config: dict of system configuration
    :return: the base model
    """

    if system_config['on_dap'] is False:
        from keras.applications import VGG16
        base_model_vgg = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(256, 256, 3) if DATA_CONFIG['resize_img'] is False else (128, 128, 3))

    if system_config['on_dap'] is True:
        with open(system_config['model_pickle_path'], 'rb') as f:
            base_model_vgg = pickle.load(f)

    return base_model_vgg


def load_model(model_config: dict, data_config: dict = DATA_CONFIG, system_config: dict = SYSTEM_CONFIG)\
        -> Union[tf.keras.Model, keras.Model]:
    """
    This function is needed because if I just declare the models (input of 'return' here) in the model_config dict
    and then call them from the train function, it doesn't work. I tried to debug this and I don't really get it, but
    it seems is it because I have to 'run' the model in the same function, cannot call the object from some other
    location?

    :param model_config: dict of model config
    :param data_config: dict of data config
    :param system_config: dict of system config
    :return: the model object
    """
    if model_config['model_name'] == 'cnn_model':
        return SimpleCnnModel(len(data_config['classes']))
    if model_config['model_name'] == 'vgg_based_model':
        return extend_pretrained_model(load_base_model(system_config), data_config)


plot_learning = PlotLearning()

early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.005,
                           patience=15, verbose=1, mode='auto')

reducelr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.001)


CNN_MODEL_CONFIG = {
    'model_name': 'cnn_model',
    'model': SimpleCnnModel(len(DATA_CONFIG['classes'])),
    'saved_weight_path': 'cnn_model',
    'train': {
        'epochs': 50,
        'optimizer': SGD(learning_rate=0.001, decay=0.001),  # also 'adam' is an option but sgd seems to work better
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'callbacks': [early_stop,
                      plot_learning,
                      reducelr]
    }
}

VGG_MODEL_CONFIG = {
    'model_name': 'vgg_based_model',
    'model': extend_pretrained_model(load_base_model(SYSTEM_CONFIG), DATA_CONFIG),
    'saved_weight_path': 'vgg_based_model',
    'train': {
        'epochs': 100,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'callbacks': [early_stop,
                      plot_learning,
                      reducelr]
    }
}
