# dataset preuzet sa https://github.com/brian-the-dev/recaptcha-dataset.git


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tensorflow.keras.utils import image_dataset_from_directory

def main():
    train_dataset, validation_dataset = load_dataset(32, (120, 120))
    show_data(train_dataset)


def load_dataset(batch_size, image_size, validation_split=0.2, seed=23):
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


def show_data(dataset, rows=3, cols=3):
    class_names = dataset.class_names

    plt.figure(figsize=(cols * 3, rows * 3))
    for images, labels in dataset.take(1):
        for i in range(rows * cols):
            ax = plt.subplot(cols, rows, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
