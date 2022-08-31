# dataset preuzet sa https://github.com/brian-the-dev/recaptcha-dataset.git


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.preprocessing import image_dataset_from_directory

def main():
    train_dataset, validation_dataset = load_dataset(32, (120, 120))


def load_dataset(batch_size, image_size, validation_split=0.2, seed=51):
    directory = "recaptcha-dataset/Large/"
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=image_size,
                                                 validation_split=validation_split,
                                                 subset='training',
                                                 seed=seed)
    validation_dataset = image_dataset_from_directory(directory,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      image_size=image_size,
                                                      validation_split=validation_split,
                                                      subset='validation',
                                                      seed=seed)
    return train_dataset, validation_dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
