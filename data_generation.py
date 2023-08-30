import os
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
    #convert 10 percent to grayscale
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        #input layer expects 3 channels -> make 3 copies of gray image
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,3])

    #modify brightness randomly
    #image = tf.image.random_brightness(image, max_delta = 0.1)
    #modify contrast randomly
    #image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    #flip images "horizontally" randomly
    image = tf.image.random_flip_left_right(image)

    return image, label


def process_image(ds, reader=None, target_size=None, augmenter=None):
    """Process images, relies on tensorflow datasets"""
    #read from files
    if not reader == None:
        ds = ds.map(reader, num_parallel_calls=AUTOTUNE)
    #normalize
    ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
    #ds = ds.cache()
    #resize
    #if not target_size == None:
    #    assert len(target_size)==2
    #    ds = ds.map(
    #            lambda img, lab: (tf.image.resize(img, target_size), lab),
    #            num_parallel_calls=AUTOTUNE)
    #perform data augmentation
    if not augmenter == None:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


class data_loader():
    def __init__(self, valid_split=0.1, augment_data=False):
        assert not (valid_split>1.0 or valid_split<0.0)
        train_split = 1.0 - valid_split

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
        self.ds_valid = self.ds_train.skip(train_size).take(valid_size)
        self.ds_train = self.ds_train.take(train_size)

        #process data
        if augment_data:
            self.ds_train = process_image(self.ds_train,
                                          augmenter=augment)
        else:
            self.ds_train = process_image(self.ds_train)
        self.ds_valid = process_image(self.ds_valid)
        self.ds_test  = process_image(self.ds_test)

    def show_examples(self):
        #Visualize data
        f, ax = plt.subplots(1, self.num_classes, figsize=(20,20))

        for i in range(0,self.num_classes):
            for image, label in self.ds_train.unbatch():
                if label == i:
                    ax[i].imshow(image*255, cmap="gray")
                    ax[i].set_title("Label: {}".format(i), fontsize=16)
                    break
            #sample = x_train[y_train==i][0]
            #ax[i].imshow(sample, cmap="gray")
            #ax[i].set_title("Label: {}".format(i), fontsize=16)

        plt.show()


class load_data_from_files():
    def __init__(self,
                 directory,
                 csv_file,
                 valid_split=0.1,
                 test_split=0.1,
                 augment_data=False):
        assert not (valid_split>1.0 or valid_split<0.0)
        assert not (test_split >1.0 or test_split <0.0)
        train_split = 1.0 - valid_split - test_split
        assert os.path.exists(directory+csv_file)

        self.directory = directory

        #Fetch data
        df = pd.read_csv(directory+csv_file)
        file_paths = df["file_name"].values
        labels     = df["label"].values

        def read_image(image_file, label):
            image = tf.io.read_file(self.directory+image_file)
            image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
            return image, label

        self.read_image = read_image
        self.input_shape = (32, 32, 3)
        self.num_classes = 10

        #convert to tensorflow datasets
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.shuffle(buffer_size = len(labels))

        #split dataset into training, validation and test data
        train_size = int(train_split*len(labels))
        valid_size = int(valid_split*len(labels))
        test_size  = int(test_split *len(labels))
        self.ds_train = dataset.take(train_size)
        self.ds_valid = dataset.skip(train_size).take(valid_size)
        self.ds_test  = dataset.skip(train_size).skip(valid_size)

        #process data
        if augment_data:
            self.ds_train = process_image(
                    self.ds_train,
                    reader=read_image,
                    augmenter=augment)
        else:
            self.ds_train = process_image(
                    self.ds_train,
                    reader=read_image)
        self.ds_valid = process_image(
                self.ds_valid,
                reader=read_image)
        self.ds_test  = process_image(
                self.ds_test,
                reader=read_image)

    def analyze_data(self, data="train"):
        if data == "train":
            labels = np.array([lab for img, lab in self.ds_train.unbatch()])
        elif data == "valid":
            labels = np.array([lab for img, lab in self.ds_valid.unbatch()])
        elif data == "test":
            labels = np.array([lab for img, lab in self.ds_test.unbatch() ])
        else:
            import sys
            sys.exit("Invalid argument")

        #create matrix
        labmat = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for i in range(len(labels)-1):
            labmat[labels[i],labels[i+1]] += 1
        labmat[labels[-1],labels[0]] += 1

        fig, ax = plt.subplots(1, 2, figsize=(10,10))
        #fig.tight_layout()
        fig.suptitle(f"Size of dataset: {len(labels)}")

        #histogram
        ax[0].hist(labels, bins=self.num_classes, density=True)
        ax[0].set_xlabel("label")
        ax[0].set_ylabel("frequency of occurence")

        #matrix
        ax[1].matshow(labmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(labmat.shape[0]):
            for j in range(labmat.shape[1]):
                ax[1].text(x=j, y=i, s=labmat[i,j], va='center', ha='center',
                        size='xx-small')
        ax[1].set_xlabel("label")
        ax[1].set_ylabel("following label in sequence")

        plt.show()

    def get_test_separate(self):
        test_images = []
        test_labels = []

        for image, label in self.ds_test.unbatch():
            test_images.append(image)
            test_labels.append(label)

        return np.array(test_images), np.array(test_labels)

    def show_examples(self):
        #Visualize data
        f, ax = plt.subplots(1, self.num_classes, figsize=(20,20))

        for i in range(0,self.num_classes):
            for image, label in self.ds_train.unbatch():
                if label == i:
                    ax[i].imshow(image*255, cmap="gray")
                    ax[i].set_title("Label: {}".format(i), fontsize=16)
                    break

        plt.show()

