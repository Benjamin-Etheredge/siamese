from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten
from .utils import bia_init, weight_init, reg

class Encoder(keras.Model):
   def __init__(self, *,
               conv_kernel_start: int = 9,
               conv_kernel_factor: int = 2,
               conv_kernel_rate: int = 2,
               conv_layers_count: int = 8,
               conv_filters_start: int = 32,
               conv_filters_factor: float = 2.,
               conv_filters_rate: int = 2,
               stride_frequency: int = 2,
               padding: str = 'same',
               conv_activation: str = 'relu',
               dense_layers_count: int = 2,
               dense_nodes: int = 1024,
               latent_nodes: int = 128,
               activation: str = 'relu',
               final_activation: str = 'relu',
               name: str = None):
      super(Encoder, self).__init__(name=name)

      # TODO should I just create the dict here?
      # setup config info
      self.conv_kernel_start = conv_kernel_start
      self.conv_kernel_factor = conv_kernel_factor
      self.conv_kernel_rate = conv_kernel_rate
      self.conv_layers_count = conv_layers_count
      self.conv_filters_start = conv_filters_start
      self.conv_filters_factor = conv_filters_factor
      self.conv_filters_rate = conv_filters_rate
      self.stride_frequency = stride_frequency
      self.padding = padding
      self.dense_layers_count = dense_layers_count
      self.dense_nodes = dense_nodes
      self.latent_nodes = latent_nodes
      self.activation = activation
      self.conv_activation = conv_activation
      self.final_activation = activation
      
      self.all_layers = []

      # TODO is it better to use class values or pass them?
      self.conv_layers = self.build_conv_layers(
               conv_kernel_rate, conv_kernel_factor, conv_kernel_rate, conv_layers_count, 
               conv_filters_start, conv_filters_factor, conv_filters_rate, 
               stride_frequency, padding, conv_activation)
      self.all_layers += self.conv_layers

      self.flattener = Flatten()
      self.all_layers.append(self.flattener)

      self.dense_layers = [
         Dense(dense_nodes, activation=activation, name=f"encoder_dense_{layer_idx}")
         for layer_idx in range(dense_layers_count)]
      self.all_layers += self.dense_layers
      
      self.latent_layer = Dense(latent_nodes, activation=final_activation)
      self.all_layers.append(self.latent_layer)
      
      self.output_dim = latent_nodes
   
   # TODO fix
   '''
   def build(self, input_shape):
      for layer in self.all_layers:
         layer.build(input_shape)
         input_shape = layer.output_shape
   '''


   # TODO just manually pass list for layer values
   @staticmethod
   def build_conv_layers(
               conv_kernel_start: int,
               conv_kernel_factor: int,
               conv_kernel_rate: int,
               conv_layers_count: int,
               conv_filters_start: int,
               conv_filters_factor: int,
               conv_filters_rate: int,
               stride_frequency: int,
               padding: str,
               conv_activation: str,
   ):
      conv_layers = []
      conv_kernel = conv_kernel_start
      conv_filters = conv_filters_start
      for layer_idx in range(1, conv_layers_count+1): # starting at one for mods below
         strides = 1 if layer_idx % stride_frequency != 0 else 2
         conv_layers.append(
            Conv2D(
               filters=conv_filters, 
               kernel_size=conv_kernel,
               strides=strides,
               padding=padding,
               activation=conv_activation,
               kernel_initializer=weight_init(), 
               bias_initializer=bia_init(), 
               kernel_regularizer=reg(),
            )
         )
         if layer_idx % conv_kernel_rate == 0:
            conv_kernel = max(1, conv_kernel - conv_kernel_factor)
         if layer_idx % conv_filters_rate == 0:
            conv_filters = int(round(conv_filters * conv_filters_factor))
      return conv_layers

   def call(self, inputs):
      x = inputs
      for layer in self.all_layers:
         x = layer(x)
      return x

   def get_config(self):
      return {
         "dense_layer": self.dense_layers,
         "dense_nodes": self.dense_nodes,
         "latent_nodes": self.latent_nodes,
         "activation": self.activation,
         "final_activation": self.final_activation,
         "name": self.name,
         "conv_kernel_start": self.conv_kernel_start,
         "conv_kernel_factor": self.conv_kernel_factor,
         "conv_kernel_rate": self.conv_kernel_rate,
         "conv_layers": self.conv_layers,
         "conv_filters_start": self.conv_filters_start,
         "conv_filters_factor": self.conv_filters_factor,
         "conv_filters_rate": self.conv_filters_rate,
         "stride_frequency": self.stride_frequency,
         "padding": self.padding,
         "conv_activation": self.conv_activation,
      }
