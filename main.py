import os
import numpy as np
from data_generation import data_loader, load_data_from_files
import models
from callbacks import make_callback_list
from trainer import model_handler
import analytics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

pdir = os.getcwd()+"/plots/"
if not os.path.exists(pdir):
    os.mkdir(pdir)

data_dir = os.getcwd()+"/data/"
assert os.path.exists(data_dir)

np.random.seed(0)

#Load data
#dataset = data_loader(valid_split=0.1, test_split=0.1, augment_data=True)

dataset = load_data_from_files(directory=data_dir, csv_file="labels.csv",
            valid_split=0.1, test_split=0.2, augment_data=False)

dataset.show_examples()

#Train model
#model = models.FNN_model_1(input_shape=input_shape)
model = models.CNN_model_1(input_shape=dataset.input_shape)
model_name = "CNN3"
prefix = model_name + "_"
wgt_path = prefix+"checkpoint/"
cma_name = pdir+prefix+"confusion_matrix.png"
err_num = 5
err_name = pdir+prefix+"top_"+str(err_num)+"_errors.png"
learning_rate = 3e-4
num_epochs = 10

handler = model_handler(model, model_name, learning_rate, pdir)

handler.set_checkpoint(wgt_path)
handler.train_model(train_data=dataset.ds_train,
                    valid_data=dataset.ds_valid,
                    epochs=num_epochs)

#Load best weights from last training
handler.load_weights(wgt_path)

handler.model.evaluate(dataset.ds_test, verbose=2)

y_pred = handler.model.predict(dataset.ds_test)

#Confusion matrix
anal = analytics.coma(dataset.ds_test, y_pred, dataset.num_classes)
anal.plot_cmat(file_name=cma_name)

#Investigate some errors
#anal.error_examples(images_from_files=True, file_name=err_name, num=5)
