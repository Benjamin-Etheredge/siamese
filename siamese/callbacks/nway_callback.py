import tensorflow as tf
import numpy as np
from copy import deepcopy

class NWayCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            encoder: tf.keras.Model, 
            head: tf.keras.Model, 
            nway_ds: tf.data.Dataset, 
            freq: int, 
            prefix_name: str = "",
            comparator = 'min',
            *args, **kwargs
    ):
        super(NWayCallback, self).__init__(*args, **kwargs)
        self.encoder = encoder # storing for faster comparisons
        self.head = head 
        self.nway_ds = nway_ds
        # TODO layout structure of nway_ds
        self.freq = freq
        self.prefix_name = prefix_name
        self.score = None
        self.avg_distance = None
        self.avg_variance = None

        if comparator == 'min':
            self.comparator = np.argmin
        elif comparator == 'max':
            self.comparator = np.argmax
        elif type(comparator) == type(int):
            # TODO Add threashold/max comparator
            raise ValueError("Future Feature")
        else:
            raise ValueError("Invalid comparator")

    # TODO pull out some of this logic
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            # TODO move to predict from comprehension
            all_encodings = [self.encoder.predict_on_batch(item) for item, _ in self.nway_ds]
            predictions = []
            avg_distances = []
            variances = []
            for encodings in all_encodings:
                anchor = encodings[0]
                anchors = tf.convert_to_tensor([anchor for _ in encodings[1:]])

                # Move expected match to prevent 100% accuracy spike when all distances are equal
                #encodings[1], encodings[2] = encodings[2], encodings[1] # TODO why does this duplicate entry?
                temp = deepcopy(encodings[2])
                encodings[2] = deepcopy(encodings[1])
                encodings[1] = temp

                distances = self.head.predict_on_batch((anchors, tf.convert_to_tensor(encodings[1:])))
                #distances = np.array([self.head((anchor, encoding)) for encoding in encodings[1:]]).flatten()
                #assert(len(distances) > 1)
                # TODO move expected value to last item since all values could be zero causing 100% accuracy due to first item being min
                
                predictions.append(self.comparator(distances))
                avg_distances.append(np.average(distances))
                variances.append(np.var(distances))

            correct_predictions = [prediction == 1 for prediction in predictions]
            self.score = np.average(correct_predictions)

            self.avg_distance = np.average(avg_distances)

            self.avg_variance = np.average(variances)
        logs[f'{self.prefix_name}nway_acc'] = self.score
        logs[f'{self.prefix_name}nway_avg_dist'] = self.avg_distance
        logs[f'{self.prefix_name}nway_avg_var'] = self.avg_variance
