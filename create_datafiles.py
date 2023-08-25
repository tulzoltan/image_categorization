import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test  = y_test.flatten()

directory = os.getcwd()+"/data/"

file = open(directory+"labels.csv","w")
file.write(f"file_name, label, extra\n")

for i in range(10):
    print(f"images in category {i}")
    for j in range(50):
        sample = x_train[y_train==i][j]
        im = Image.fromarray(sample)
        name = "IMG"+str(i)+"_"+str(j)
        im.save(directory+name+".jpg")
        file.write(f"{name}, {i}, 0\n")

file.close()
