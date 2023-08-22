import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import models
from callbacks import make_callback_list


class model_handler():
    def __init__(self, model, model_name, learning_rate, save_location):
        self.model = model

        #compile model
        self.model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
        )

        self.save_dir = save_location
        self.prefix = model_name + "_"

        self.save = False

    def set_checkpoint(self, wgt_path):
        save_checkpoint = keras.callbacks.ModelCheckpoint(
                wgt_path,
                save_weights_only=True,
                monitor="accuracy",
                save_best_only=True,
                )

        self.my_callback = make_callback_list()
        self.my_callback.insert(0,save_checkpoint)
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

