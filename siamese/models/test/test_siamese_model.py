import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from .. import SiameseModel

   
@pytest.fixture(scope="session")
def generated_data(item_count=10, label_count=11, feature_count=3):
   data = np.array([np.random.rand(feature_count) for _ in range(item_count)])
   labels = [idx%label_count for idx in range(item_count)]
   return data, labels



@pytest.fixture(scope="session")
def model():
   encoder = tf.keras.models.Sequential([
      Dense(32),
      Dense(32)
   ])
   head = tf.keras.models.Sequential([
      Dense(32),
      Dense(1)
   ])
   model = SiameseModel(
      encoder_model=encoder,
      head_model=head
   )
   return model

def test_init():
   encoder = tf.keras.models.Sequential([
      Dense(32),
      Dense(32)
   ])
   head = tf.keras.models.Sequential([
      Dense(32),
      Dense(32)
   ])
   model = SiameseModel(
      encoder_model=encoder,
      head_model=head
   )

   assert model.encoder == encoder
   assert model.encoder.get_config() == encoder.get_config()

   assert model.head == head
   assert model.head.get_config() == head.get_config()

def test_save_load(model, generated_data, tmpdir_factory):
   fn = tmpdir_factory.mktemp("data")
   model_path = fn.join("model")
   assert not model_path.exists()
      
   data, labels = generated_data
   with pytest.raises(ValueError):
      model.save(model_path)
   assert not model_path.exists()

   #model.build(np.shape(data[0]))
   model.predict([data[0:], data[0:]])
   model.summary()
   #model(generated_data[0]) # build model since .build doesn't work how I want yet
   model.save(model_path)
   assert model_path.exists()

   loaded_model = tf.keras.models.load_model(model_path)
   for sub_model, loaded_sub_model in zip(model.layers, loaded_model.layers):
      for layer, loaded_layer in zip(sub_model.layers, loaded_sub_model.layers):
         assert layer.get_config() == loaded_layer.get_config()