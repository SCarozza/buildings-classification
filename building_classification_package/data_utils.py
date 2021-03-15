from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator as kerasDirectoryIterator

from building_classification_package.config import DATA_CONFIG


def get_data_dir(which_set: str) -> str:
    """
    This function is to specifically extract data from a folder in which train and test are split, like the data
    I used to train this model. The two train and test folders are called Building_labeled_train_data and Building_labeled_test_data,
    and thy each contain 8 subfolders with the images corresponding to each of the 8 classes.
    :param which_set: 'train' or 'test'
    :return: string of path
    """
    return f"../data/Building_labeled_{which_set}_data"


def build_dataset(set_to_build: str,
                  dataset_path: str,
                  validation_split: float = 0.0,
                  data_config: dict = DATA_CONFIG,
                  seed: int = 42,
                  **augmentation_params
                  ) -> kerasDirectoryIterator:
    """
    Builds the set we need (train, validation or test). the sets are iterator, and the starting point are images
    split in a folder for each class.

    :param set_to_build: 'train', 'val' or 'test'
    :param dataset_path: path to data
    :param validation_split: if we are building training or validation, specify the split. default is 0.0 so it works
    for building test set
    :param data_config: dict of configuration for data, such as classes to use, preprocessing_function, resize
    :param seed: random seed
    :param augmentation_params: if we want to perform data augmentation, specify params here
    :return: iterator for the desired set
    """

    target_size = (128, 128) if data_config['resize_img'] else (256, 256)

    if set_to_build not in ['train', 'test', 'val', 'validation']:
        raise TypeError("Set must be train, test or val")

    split_subset = 'training' if set_to_build is 'train' \
        else 'validation' if set_to_build in ['val', 'validation'] \
        else None

    # 1. define the generator:
    datagenerator = ImageDataGenerator(preprocessing_function=data_config['preprocessing_fuction'],
                                       validation_split=validation_split, rescale=1/256,
                                       **augmentation_params
                                       )

    # 2. import data using the generator
    dataset_iterator = datagenerator.flow_from_directory(
        dataset_path,
        classes=data_config['classes'],
        target_size=target_size,
        subset=split_subset,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
        batch_size=32
    )
    assert dataset_iterator.next() is not None

    return dataset_iterator
