import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import models


def scheduler99(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr*0.99


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.975:
            self.model.stop_training = True


class model_handler():
    def __init__(self, model, model_name, initial_lr, save_location, decay_steps=1):
        self.model = model

        #learning rate scheduler
        self.learning_rate = keras.optimizers.schedules.CosineDecayRestarts(initial_lr, decay_steps)

        #compile model
        self.model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        metrics=["accuracy"],
        )

        self.save_dir = save_location
        self.prefix = model_name + "_"

        self.save = False

    def set_checkpoint(self, wgt_path):
        #lr_scheduler = LearningRateScheduler(scheduler99, verbose=1)
        lr_scheduler = LearningRateScheduler(self.learning_rate, verbose=1)

        save_checkpoint = keras.callbacks.ModelCheckpoint(
                wgt_path,
                save_weights_only=True,
                monitor="accuracy",
                save_best_only=True,
                )

        #self.my_callback = [save_checkpoint]
        self.my_callback = [save_checkpoint, lr_scheduler, CustomCallback()]
        self.save = True

    def load_weights(self, wgt_path):
        if os.path.exists(wgt_path):
            self.model.load_weights(wgt_path)

    def train_model(self, train_data, valid_data, epochs):
        if self.save:
            history = self.model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=2,
                    callbacks=self.my_callback
                    )
        else:
            history = self.model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=2
                    )

        hdf = pd.DataFrame.from_dict(data=history.history, orient='columns')
        hdf.to_csv(self.save_dir+self.prefix+"history.csv", header=True, index=False)

