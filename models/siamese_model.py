import tensorflow as tf

class SiameseModel(tf.keras.Model):
   def __init__(
         self, 
         encoder_model: tf.keras.Model, 
         head_model: tf.keras.Model, 
         name=None):
      super(SiameseModel, self).__init__(name=name)
      self._encoder = encoder_model
      self._head = head_model
      # TODO do I need to pass through the args?  can do en_kwargs/h_kwargs

   def call(self, input):
      anchor, other = input
      return self.head(self.encoder(anchor), self.encoder(other))

   def get_config(self):
      return {
         "encoder_model": {"class_name": type(self.encoder).__name__,
                           "config": self.encoder.get_config()},
         "head_model": {"class_name": type(self.head).__name__,
                           "config": self.head.get_config()},
      }

   @property
   def encoder(self):
      return self._encoder

   @property
   def head(self):
      return self._head
# TODO figure out custom object  handling
#CUSTOM_OBJECTS = {"Encoder": Encoder, "NormDistanceLayer": NormDistanceLayer}
   