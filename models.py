import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

#Create sequential FNN model
def FNN_model_1(input_shape):
    fnn = keras.models.Sequential([
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.25),
            layers.Dense(10, activation='softmax')
        ])
    return fnn


#Create functional CNN model
def CNN_model_1(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class RandomFlip(layers.Layer):
    def __init__(self, mode):
        super(RandomFlip, self).__init__()
        self.mode = mode

    def call(self, image, training=None):
        if not training:
            return image
        if self.mode == "horizontal":
            image = tf.image.random_flip_left_right(image)
        elif self.mode =="vertical":
            image = tf.image.random_flip_up_down(image)
        return image


class RandomContrast(layers.Layer):
    def __init__(self, lower, upper):
        super(RandomContrast, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self, images, training=None):
        if not training:
            return images
        #images = tf.image.random_contrast(images, lower=self.lower,
        #                            upper=self.upper)
        contrast = np.random.uniform(self.lower, self.upper)
        images = tf.image.adjust_contrast(images, contrast)
        images = tf.clip_by_value(images, 0, 1)
        return images


class RandomBrightness(layers.Layer):
    def __init__(self, max_delta):
        super(RandomBrightness, self).__init__()
        self.max_delta = max_delta

    def call(self, image):
        image = tf.image.random_brightness(image, max_delta=self.max_delta)
        return image


class RandomGrayscale(layers.Layer):
    def __init__(self, prob):
        super(RandomGrayscale, self).__init__()
        self.prob = prob

    def call(self, image):
        if tf.random.uniform((), minval=0, maxval=1) < self.prob:
            image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,1,3])
        return image


def augment(seq):
    #seq = layers.Resizing(height=32, width=32)(seq)
    #seq = RandomFlip(mode="horizontal")(seq)
    #seq = RandomContrast(lower=0.1, upper=0.2)(seq)
    #seq = RandomBrightness(max_delta=0.1)(seq)
    #seq = RandomGrayscale(prob=0.1)(seq)
    return seq


#Create functional CNN model
def CNN_model_2(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = augment(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3,
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

