import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import models
from callbacks import make_callback_list
from trainer import model_handler
import analytics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

pdir = "plots/"
if not os.path.exists(pdir):
    os.mkdir(pdir)

np.random.seed(0)

#Fetch data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

#Visualize data
#input_shape = x_train.shape[1:] + (1,) #for mnist
input_shape = x_train.shape[1:] #for cifar10
num_classes = 10

"""
f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0,num_classes):
    sample = x_train[y_train==i][0]
    ax[i].imshow(sample, cmap="gray")
    ax[i].set_title("Label: {}".format(i), fontsize=16)

plt.show()
"""

#Prepare data
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


#convert to tensorflow datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test  = tf.data.Dataset.from_tensor_slices((x_test , y_test ))

#split training data intoo training and validation data
train_size = int(0.875*len(x_train))
ds_valid = ds_train.skip(train_size)
ds_train = ds_train.take(train_size)

#process training data
ds_train = ds_train.map(normalize, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_valid = ds_valid.map(normalize, num_parallel_calls=AUTOTUNE)
ds_valid = ds_valid.batch(BATCH_SIZE)
ds_valid = ds_valid.prefetch(AUTOTUNE)

#process test data
ds_test  = ds_test.map(normalize, num_parallel_calls=AUTOTUNE)
ds_test  = ds_test.batch(BATCH_SIZE)
ds_test  = ds_test.prefetch(AUTOTUNE)

x_test = x_test.astype("float32") / 255.0

#Train
#model = models.FNN_model_1(input_shape=input_shape)
model = models.CNN_model_1(input_shape=input_shape)
model_name = "CNN2"
prefix = model_name + "_"
wgt_path = prefix+"checkpoint/"
cma_name = pdir+prefix+"confusion_matrix.png"
err_num = 5
err_name = pdir+prefix+"top_"+str(err_num)+"_errors.png"
learning_rate = 3e-4
num_epochs = 20

handler = model_handler(model, model_name, learning_rate, pdir)
handler.set_checkpoint(wgt_path)
handler.train_model(train_data=ds_train,
                    valid_data=ds_valid,
                    epochs=num_epochs)

#Load best weights from last training
handler.load_weights(wgt_path)

handler.model.evaluate(ds_test, verbose=2)

#Confusion matrix
anal = analytics.coma(handler.model, x_test, y_test, num_classes)
anal.plot_cmat(file_name=cma_name)

#INVESTIGATE SOME ERRORS
anal.error_examples(file_name=err_name, num=5)
