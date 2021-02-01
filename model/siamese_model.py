import tensorflow as tf
from tensorflow import keras
from model.encoders import Encoder


class SiameseModel(keras.Model):
   def __init__(self, encoder_model, head_model, name=None):
      super(SiameseModel, self).__init__(name=name)
      self.encoder = encoder_model
      self.head = head_model
      # TODO do I need to pass through the args?  can do en_kwargs/h_kwargs

   def get_config(self):
      return {
         "encoder_model": selfencoder_model,
         "head_model": selfhead_modelS,
      }

# TODO figure out custom object  handling
#CUSTOM_OBJECTS = {"Encoder": Encoder, "NormDistanceLayer": NormDistanceLayer}
   