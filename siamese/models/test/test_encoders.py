from .. import encoders
import pytest
import numpy as np

@pytest.mark.parametrize("conv_count", range(1,10, 3))
@pytest.mark.parametrize("dense_count", range(1,10, 3))
def test_layer_counts(conv_count, dense_count):

   encoder = encoders.Encoder(conv_layers_count=conv_count, dense_layers_count=dense_count)
   assert encoder.conv_layers_count == conv_count
   assert encoder.dense_layers_count == dense_count
   layer_count = len(encoder.layers)
   # addition 1 for latent layer, 1 for flatten
   assert layer_count == conv_count + dense_count + 2, \
         f"Incorrect Layer count: {encoder.layers}"


import tensorflow as tf
@pytest.mark.parametrize("input_shape", [
      #(1,), (10,), (200,), (4444,),
      #(1, 1), 
      (1, 1, 1),
      (224, 224, 3), 
      (100, 100, 100)])
def test_encoder_build(input_shape):
   encoder = encoders.Encoder()

   # model shouldn't be built yet so inputs can vary
   with pytest.raises(ValueError):
      encoder.summary()

   #try:
   # Build seems to be broken...
   #encoder.build([None, *input_shape])
   data = np.random.rand(1, 225, 225, 3) * 255
   encoder(data)
   encoder.summary()
   #except:
      #pytest.fail("Unexpected Error")


#@pytest.fixture(scope="session")
def test_save_load(tmpdir):
   encoder = encoders.Encoder()
   
   #encoder.build(input_shape=(100, 100, 3))
   data = np.random.rand(1, 225, 225, 3) * 255
   encoder(data)

   model_file = tmpdir.mkdir("data").join("model")
   encoder.save(model_file)

   loaded_encoder = tf.keras.models.load_model(model_file)
   for layer, loaded_layer in zip(encoder.layers, loaded_encoder.layers):
      assert layer.get_config() == loaded_layer.get_config()

   #encoder.save(file_path, save_format='tf')