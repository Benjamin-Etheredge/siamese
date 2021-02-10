import tensorflow as tf
import numpy as np

class NWayCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            encoder: tf.keras.Model, 
            head: tf.keras.Model, 
            nway_ds: tf.data.Dataset, 
            freq: int, 
            *args, **kwargs):
        super(NWayCallback, self).__init__(*args, **kwargs)
        self.encoder = encoder # storing for faster comparisons
        self.head = head 
        self.nway_ds = nway_ds
        # TODO layout structure of nway_ds
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            all_encodings = [self.encoder.predict_on_batch(item) for item, _ in self.nway_ds]
            assert(len(all_encodings) > 1)
            predictions = []
            distances = []
            for encodings in all_encodings:
                assert(len(encodings) > 1)
                anchor = encodings[0]

                distances = np.array([self.head((anchor, encoding)) for encoding in encodings[1:]]).flatten()
                assert(len(distances) > 1)
                
                predictions.append(np.argmin(distances))

            correct_predictions = [prediction == 0 for prediction in predictions]
            score = np.average(correct_predictions)
            logs['nway_acc'] = score

