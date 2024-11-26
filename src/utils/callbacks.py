import tensorflow as tf
import os

class Callbacks:
    @staticmethod
    def create_checkpoint_callback(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            mode='min',
            verbose=0,
            monitor='val_mean_absolute_error'
        )
