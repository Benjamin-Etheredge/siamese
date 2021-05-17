import tensorflow as tf
from pathlib import Path
import os
from types import FunctionType


@tf.function
def bool_mask(item, items):
   '''Function for producing/testing a tf bool mask'''
   return item == items


@tf.function
def get_pair(
         item_keys, 
         labels, 
         anchor_key, 
         anchor_label, 
         output_label=None
):
   '''Get pairwise data'''

   tf.debugging.assert_equal(len(tf.shape(item_keys)), tf.constant(1))
   tf.debugging.assert_equal(len(tf.shape(labels)), tf.constant(1))

   if output_label is None:
      output_label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)

   anchor_mask = bool_mask(anchor_key, item_keys)

   pos_mask = bool_mask(anchor_label, labels)
   tf.debugging.assert_equal(tf.size(pos_mask), tf.size(labels), f"mask is wrong.\n")
   tf.debugging.assert_less(tf.constant(1), tf.size(labels), f"Split is wrong.\n")
   tf.debugging.assert_equal(len(tf.shape(pos_mask)), tf.constant(1), "mask shape is wrong.\n")

   pos_label_func = lambda: tf.math.logical_xor(pos_mask, anchor_mask)  # XOR prevents anchor file being used
   neg_label_func = lambda: tf.math.logical_not(pos_mask)
   mask = tf.cond(output_label == tf.constant(1), pos_label_func, neg_label_func)
   tf.debugging.assert_equal(len(tf.shape(mask)), tf.constant(1))
   values = tf.boolean_mask(item_keys, mask)

   tf.debugging.assert_greater(tf.size(values), tf.constant(0), f"Values are empty.\n")
   idx = tf.random.uniform([], 0, tf.size(values), dtype=tf.int32)
   value = tf.gather(values, idx)
   path = value
   sq_path = tf.squeeze(path)
   return anchor_key, sq_path, output_label


def create_dataset(
      anchor_items: tf.Tensor, 
      anchor_labels: tf.Tensor,
      anchor_decode_func: FunctionType = lambda x: x,
      other_items: tf.Tensor = None, 
      other_labels: tf.Tensor = None,
      other_decode_func: FunctionType = None,
      repeat: int = None
) -> tf.data.Dataset:

   if other_items is not None or other_labels is not None:
      assert (other_items is not None and other_labels is not None), "invalid others"
      items = tf.concat((anchor_items, other_items), axis=0)
      labels = tf.concat((anchor_labels, other_labels), axis=0)
   else:
      items = anchor_items
      labels = anchor_labels

   item_count = int(tf.size(items))
   assert item_count > 0, "No items found"

   label_count = int(tf.size(labels))
   assert label_count > 0, "No labels found"

   assert label_count == item_count, "Count of labels and items don't match"

   item_ds = tf.data.Dataset.from_tensor_slices(anchor_items)
   label_ds = tf.data.Dataset.from_tensor_slices(labels)
   ds = tf.data.Dataset.zip((item_ds, label_ds))
   ds = ds.cache()

   ds = ds.shuffle(item_count, reshuffle_each_iteration=True) # TODO pass seed? does that make it deterministic

   # TODO add filter to throw out items with less than 2 examples
   def parier(item, label): # testing switch away from lambda due to tensorflow graph error using lambdas
      return get_pair(items, labels, item, label)

   ds = ds.map(parier, num_parallel_calls=-1, deterministic=False)

   if repeat:
      ds = ds.repeat(repeat)

   decoder_func = decoder_builder(anchor_decode_func, other_decode_func)
   ds = ds.map(decoder_func, num_parallel_calls=-1, deterministic=False)
   
   ds = ds.prefetch(-1)
   
   return ds 


def decoder_builder(
         anchor_decoder: FunctionType, 
         other_decoder: FunctionType = None
) -> tf.data.Dataset:

   if other_decoder is None:
      other_decoder = anchor_decoder

   @tf.function
   def decode(anchor, other, label):
      return (anchor_decoder(anchor), other_decoder(other)), label

   return decode


def create_decoder(anchor_decoder: FunctionType,
                   other_decoder: FunctionType = None) -> FunctionType:
   if other_decoder is None:
      other_decoder = anchor_decoder
   
   return lambda anchor, other, label: (anchor_decoder(anchor), other_decoder(other), label)
