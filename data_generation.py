import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64


def normalize(image, label):
    """Normalize image"""
    return tf.cast(image, tf.float32)/255.0, label

def augment(image, label):
    """Data augmentation"""
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


def process_image(ds, reader=None, augmenter=None):
    """Process images, relies on tensorflow datasets"""
    if not reader == None:
        ds = ds.map(reader, num_parallel_calls=AUTOTUNE)
    ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if not augmenter == None:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


class data_loader():
    def __init__(self, valid_split=0.1, test_split=0.1, augment_data=False):
        assert not (valid_split>1.0 or valid_split<0.0)
        assert not (test_split >1.0 or test_split <0.0)
        train_split = 1.0 - valid_split - test_split

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

        #split dataset into training, validation and test data
        train_size = int(train_split*len(x_train))
        valid_size = int(valid_split*len(x_train))
        test_size  = int(test_split *len(x_train))
        self.ds_valid = self.ds_train.skip(train_size).take(valid_size)
        self.ds_train = self.ds_train.take(train_size)

        #process data
        if augment_data:
            self.ds_train = process_image(self.ds_train, augmenter=augment)
        else:
            self.ds_train = process_image(self.ds_train)
        self.ds_valid = process_image(self.ds_valid)
        self.ds_test  = process_image(self.ds_test)


class load_data_from_files():
    def __init__(self, directory, valid_split=0.1, test_split=0.1, augment_data=False):
        assert not (valid_split>1.0 or valid_split<0.0)
        assert not (test_split >1.0 or test_split <0.0)
        train_split = 1.0 - valid_split - test_split

        #Fetch data
        df = pd.read_csv(directory+file_name)
        file_paths = df['file_name'].values
        labels     = df['label'].values

        def read_image(image_file, label):
            image = tf.io.read_file(directory+image_file)
            image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
            return image, label

        self.input_shape = (32, 32, 3)
        self.num_classes = 10

        #convert to tensorflow datasets
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        #split dataset into training, validation and test data
        train_size = int(train_split*len(labels))
        valid_size = int(valid_split*len(labels))
        test_size  = int(test_split *len(labels))
        self.ds_train = dataset.take(train_size)
        self.ds_valid = dataset.skip(train_size).take(valid_size)
        self.ds_test  = dataset.skip(train_size).skip(valid_size)

        #process data
        process_image(self.ds_train, reader=read_image, augmenter=augment)
        process_image(self.ds_valid, reader=read_image)
        process_image(self.ds_test , reader=read_image)

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
