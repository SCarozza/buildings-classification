.. Buildings classification documentation master file, created by
   sphinx-quickstart on Fri Sep 25 15:45:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Buildings classification's documentation!
====================================================

This project's aim is to classify buildings based on facades images (usually from google maps). Examples of classes
we want to predict are: house, apartment, industrial, office building, church, retail.

This is a stretch ambition project, and it is inspired by:

1. the struggle of KYC analysts who, among other things, often have to check on google maps the address of companies to
verify if the building 'makes sense' (e.g. is the customer registered as an industrial activity but the address correspond to
a private house or apartment?)

2. this `paper <https://www.researchgate.net/publication/322168840_Building_Instance_Classification_Using_Street_View_Images>`_ where CNN
are used to predict classes of buildings from Google street images.


The code used to extract the data, run and evaluate the model and generate prediction is in the codebase (*building_classification_package*).
I used a notebook to run the code and analyze the results, which you can find in the *notebooks* folder.


Find in the attached powerpoint details about data, models and results.


Note on the Data
-------

To train the models I used the same dataset used in the paper: download the data at this `link <ds%2FBIC_GSV.tar.gzhttps://www.researchgate.net/deref/http%3A%2F%2Fwww.sipeo.bgu.tum.de%2Fdownloa>`_.
Download the data, unzip them, and move them in a folder called *data* inside the root of the project (or change accordingly the path in the notebook).


Note on the Models
------

I developed two different CNN models:

1. a simple CNN model which I built from scratch (with keras) with 2 convolution blocks

2. a model based on a pre-trained VGG16 network provided by Keras (transfer learning).

Both models can be modified from the *config* file in the codebase, or a different
base model (e.g. inception, or VGG19) can be used for transfer learning. Note that if you want to use a different
base model, you should probably adjust the data preprocessing accordingly. At the moment the data preprocessing is performed
using the preprocessing function used for VGG (provided by Keras). This setting can be changed in the *config* file.


Note for usage on DAP
------------

The package is made so it is possible to run it either locally on your laptop or on DAP. For training the model, it is
recommended to have GPUs so better to run on DAP. However, as on DAP you are not connected to the internet, you cannot
download the pre-trained VGG model while running. Locally, running the import statement for VGG is enough, but on DAP
I manually loaded a pickle file containing the model which I had previously imported locally.

Run the following locally to get the pickle you will need in DAP:

.. code-block:: python

   from keras.applications.vgg16 import VGG16
   base_model_vgg256 = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
   with open(f'base_model_vgg256.pickle', 'wb') as f:
       pickle.dump(base_model_vgg256, f)



On DAP you can use the venv I created for this project (*image_classification*), or create your own using the requirements.txt file.

Modify the *SYSTEM_CONFIG* settings in the *config* file
to run on DAP: set 'on_dap' to True, and add the path to the pickle file of the pre-downloaded VGG model.



Codebase
=======

Config
-----


.. automodule:: building_classification_package.config
    :members:



Model Utils
----


.. automodule:: building_classification_package.model_utils
    :members:



Data Utils
---


.. automodule:: building_classification_package.data_utils
    :members:


Callbacks
---


.. automodule:: building_classification_package.callbacks
    :members:




.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
