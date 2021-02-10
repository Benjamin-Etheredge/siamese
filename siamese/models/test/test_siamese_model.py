import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras import Model
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
   inputs = (Input(32), Input(32))
   x = Dense(1)(Dense(32)(Concatenate()(inputs)))
   head = Model(inputs=inputs, outputs=x)
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

from ..siamese_model import create_siamese_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Input, Model
# TODO convert to mock
def test_init_func():
   input_e = Input(32)
   encoder = Model(inputs=input_e, outputs=Dense(32)(Dense(32)(input_e)))
   input_h = (Input(32), Input(32))
   head = Model(inputs=input_h, outputs=Dense(16)(Dense(16)(Concatenate()(input_h))))
   model = create_siamese_model(
      encoder=encoder,
      head=head
   )


   #assert model.layers[0]== encoder
   #assert model.layers[0].get_config() == encoder.get_config()

   #assert model.layers[1] == head
   # TODO figure out a better way to test this
   #assert model.layers[1].get_config() == head.get_config()

'''
def test_save_load_func(model, generated_data, tmpdir_factory):
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
'''