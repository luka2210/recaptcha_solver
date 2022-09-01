# dataset preuzet sa https://github.com/brian-the-dev/recaptcha-dataset.git


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    batch_size = 128
    image_size = (120, 120)
    train_dataset, validation_dataset = load_dataset("recaptcha-dataset/Large/", batch_size, image_size)

    show_data(train_dataset)

    input_shape = image_size + (3,)
    recaptcha_model = build_recaptcha_model(input_shape)

    learning_rate = 0.0001
    recaptcha_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
    recaptcha_model.summary()

    epochs = 20
    history = recaptcha_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
    show_accuracy(history)


def build_recaptcha_model(input_shape):
    efficient_net = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=input_shape,
                                                                          include_top=False,
                                                                          weights='imagenet',
                                                                          classes=12)
    efficient_net.trainable = True
    fine_tune_at = len(efficient_net.layers) - 50
    for layer in efficient_net.layers[:fine_tune_at]:
        layer.trainable = False

    model = tf.keras.models.Sequential([
        data_augmentation(input_shape),
        efficient_net,
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(320, activation='leaky_relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(80, activation='leaky_relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(9, activation='softmax'),
    ])
    return model


def data_augmentation(input_shape):
    augmentation_model = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal', input_shape=input_shape),
        tf.keras.layers.RandomRotation(0.15),
        # tf.keras.layers.RandomBrightness(0.2),
        # tf.keras.layers.RandomContrast(0.2)
    ])
    return augmentation_model


def show_accuracy(history):
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def load_dataset(directory, batch_size, image_size, validation_split=0.2, seed=23):
    train_dataset1 = tf.keras.utils.image_dataset_from_directory(directory,
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
    return train_dataset1, validation_dataset


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
