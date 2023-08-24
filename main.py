import os
import numpy as np
from data_generation import data_loader
import models
from callbacks import make_callback_list
from trainer import model_handler
import analytics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

pdir = os.getcwd()+"plots/"
if not os.path.exists(pdir):
    os.mkdir(pdir)

np.random.seed(0)

dataset = data_loader(valid_split=0.1, test_split=0.1, augment_data=True)

#Train model
#model = models.FNN_model_1(input_shape=input_shape)
model = models.CNN_model_1(input_shape=dataset.input_shape)
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
handler.train_model(train_data=dataset.ds_train,
                    valid_data=dataset.ds_valid,
                    epochs=num_epochs)

#Load best weights from last training
handler.load_weights(wgt_path)

handler.model.evaluate(dataset.ds_test, verbose=2)

y_pred = handler.model.predict(dataset.ds_test)

test_images = []
test_labels = []
for image, label in dataset.ds_test.unbatch().as_numpy_iterator():
    test_images.append(image)
    test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Confusion matrix
anal = analytics.coma(test_images, test_labels, y_pred, dataset.num_classes)
anal.plot_cmat(file_name=cma_name)

#INVESTIGATE SOME ERRORS
anal.error_examples(file_name=err_name, num=5)
