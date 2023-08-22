import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns


class coma():
    def __init__(self, model, x_test, y_test, num_classes):
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        #errors
        errors = (y_pred_classes - y_true != 0)
        self.y_pred_classes_errors = y_pred_classes[errors]
        self.y_pred_errors = y_pred[errors]
        self.y_true_errors = y_true[errors]
        self.x_test_errors = x_test[errors]

        #Confusion matrix
        self.conf_mat = confusion_matrix(y_true, y_pred_classes)

    def plot_cmat(self, file_name=None):
        fig, ax = plt.subplots(figsize=(15,10))
        ax = sns.heatmap(self.conf_mat, annot=True, fmt='d', ax=ax,
                         cmap="Blues")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion matrix")
        if file_name!=None:
            if not os.path.exists(file_name):
                plt.savefig(file_name)
        plt.show()

    def error_examples(self, file_name=None, num=5):
        y_pred_errors_probability = np.max(self.y_pred_errors, axis=1)
        true_probability_errors = np.diagonal(np.take(self.y_pred_errors, self.y_true_errors, axis=1))
        diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

        #get list of indices of sorted differences
        sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
        top_idx_diff_errors = sorted_idx_diff_errors[-num:]

        f, ax = plt.subplots(1, num, figsize=(30,30))

        for i in range(0,num):
            idx = top_idx_diff_errors[i]
            sample = self.x_test_errors[idx]
            y_t = self.y_true_errors[idx]
            y_p = self.y_pred_classes_errors[idx]
            ax[i].imshow(sample, cmap = 'gray')
            ax[i].set_title("Predicted label: {}\nTrue label: {}".format(y_p, y_t), fontsize=22)

        if file_name!=None:
            if not os.path.exists(file_name):
                plt.savefig(file_name)
        plt.show()
