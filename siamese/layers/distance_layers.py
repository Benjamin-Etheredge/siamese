import tensorflow as tf

class NormDistanceLayer(tf.keras.layers.Layer):
   def __init__(self, **kwargs):
      super(NormDistanceLayer, self).__init__(**kwargs)

   def call(self, inputs):
      x, y = inputs
      return tf.norm(x-y, axis=-1, keepdims=True)