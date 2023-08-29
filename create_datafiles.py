import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test  = y_test.flatten()

directory = os.getcwd()+"/data/"

file = open(directory+"labels.csv","w")
file.write(f"file_name,label\n")

num_classes = 10
images_per_class = 5000
num_images = images_per_class*num_classes

print(f"Saving {num_images} out of {len(x_train)} images")

for i in range(num_classes):
    print(f"images in category {i}")
    for j in range(images_per_class):
        sample = x_train[y_train==i][j]
        im = Image.fromarray(sample)
        name = "IMG"+str(i)+"_"+str(j)+".jpg"
        im.save(directory+name)
        file.write(f"{name},{i}\n")

file.close()
