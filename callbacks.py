from tensorflow.keras.callbacks import Callback, LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr*0.99

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.975:
            self.model.stop_training = True

def make_callback_list():
    return [lr_scheduler, CustomCallback()]
