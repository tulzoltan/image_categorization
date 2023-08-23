import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

def normalize(image, label):
    return tf.cast(image, tf.float32)/255.0, label

def augment(image, label):
    #new_hgt = new_wdt = 32
    #image = tf.image.resize(image, (new_hgt, new_wdt))

    #convert 10 percent to grayscale
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        #input layer expects 3 channels -> make 3 copies of gray image
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,3])

    #modify brightness randomly
    #image = tf.image.random_brightness(image, max_delta = 0.1)
    #modify contrast randomly
    #image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    #flip images "horizontall" randomly
    image = tf.image.random_flip_left_right(image)

    return image, label


class data_loader():
    def __init__(self, augment_data=False):
        #Fetch data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        #self.input_shape = x_train.shape[1:] + (1,) #for mnist
        self.input_shape = x_train.shape[1:] #for cifar10
        self.num_classes = 10

        #convert to tensorflow datasets
        self.ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.ds_test  = tf.data.Dataset.from_tensor_slices((x_test , y_test ))

        #split training data intoo training and validation data
        train_size = int(0.875*len(x_train))
        self.ds_valid = self.ds_train.skip(train_size)
        self.ds_train = self.ds_train.take(train_size)

        #process training data
        self.ds_train = self.ds_train.map(normalize, num_parallel_calls=AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        if augment_data:
            self.ds_train = self.ds_train.map(augment, num_parallel_calls=AUTOTUNE)
        self.ds_train = self.ds_train.batch(BATCH_SIZE)
        self.ds_train = self.ds_train.prefetch(AUTOTUNE)

        self.ds_valid = self.ds_valid.map(normalize, num_parallel_calls=AUTOTUNE)
        self.ds_valid = self.ds_valid.batch(BATCH_SIZE)
        self.ds_valid = self.ds_valid.prefetch(AUTOTUNE)

        #process test data
        self.ds_test  = self.ds_test.map(normalize, num_parallel_calls=AUTOTUNE)
        self.ds_test  = self.ds_test.batch(BATCH_SIZE)
        self.ds_test  = self.ds_test.prefetch(AUTOTUNE)

"""
    def show_examples(self):
        #Visualize data
        f, ax = plt.subplots(1, num_classes, figsize=(20,20))

        for i in range(0,num_classes):
            sample = x_train[y_train==i][0]
            ax[i].imshow(sample, cmap="gray")
            ax[i].set_title("Label: {}".format(i), fontsize=16)

        plt.show()
"""
