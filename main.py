# dataset preuzet sa https://github.com/brian-the-dev/recaptcha-dataset.git


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    batch_size = 32
    image_size = (120, 120)
    train_dataset, validation_dataset = load_dataset("recaptcha-dataset/Large/", batch_size, image_size)

    show_data(train_dataset)

    input_shape = image_size + (3,)
    recaptcha_model = build_recaptcha_model(input_shape)
    recaptcha_model.summary()

    learning_rate = 0.001
    recaptcha_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])


def build_recaptcha_model(input_shape):
    efficient_net = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=input_shape,
                                                                          include_top=False,
                                                                          weights='imagenet',
                                                                          classes=12)
    model = tf.keras.models.Sequential([
        # tf.keras.layers.InputLayer(input_shape=input_shape),
        efficient_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])
    return model


def load_dataset(directory, batch_size, image_size, validation_split=0.2, seed=23):
    train_dataset = tf.keras.utils.image_dataset_from_directory(directory,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=image_size,
                                                                validation_split=validation_split,
                                                                subset='training',
                                                                seed=seed)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(directory,
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
