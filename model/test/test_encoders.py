from model import encoders
import pytest
from icecream import ic

@pytest.mark.parametrize("conv_count", range(1,10, 3))
@pytest.mark.parametrize("dense_count", range(1,10, 3))
def test_layer_counts(conv_count, dense_count):

   encoder = encoders.Encoder(conv_layers_count=conv_count, dense_layers_count=dense_count)
   assert encoder.conv_layers_count == conv_count
   assert encoder.dense_layers_count == dense_count
   layer_count = len(encoder.layers)
   assert layer_count == conv_count + dense_count + 1 # addition 1 for latent layer

import tensorflow as tf
#@pytest.fixture(scope="session")
def test_save_load(tmpdir):
   encoder = encoders.Encoder()
   
   with pytest.
   encoder.summary()

   model_file = tmpdir.mkdir("data").join('model')
   encoder.save(model_file)

   loaded_encoder = tf.keras.models.load_model(model_file)
   #ic(model)
   for layer, loaded_layer in zip(encoder.layers, loaded_encoder):
      pass
   assert loaded_encoder == encoder

   #encoder.save(file_path, save_format='tf')

def test_test():
   pass
   #assert False

