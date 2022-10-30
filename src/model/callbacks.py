import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from src.model.model_handler import ModelHandler

from src.model.config import *

class SaveUpdateStepCallback(Callback):

    def __init__(self, model_handler):

        super(SaveUpdateStepCallback, self).__init__()

        self.model_handler = model_handler

    def on_epoch_end(self, epoch, logs=None):

        self.model_handler.update_current_step()

        if (self.model_handler.current_step) % EPOCHS_BEFORE_SAVE == 0:

            self.model_handler.optimizer.swap_weights()

            if SAVE_MODEL:
                self.model_handler.save_weights()

            self.model_handler.save_samples()

            self.model_handler.optimizer.swap_weights()