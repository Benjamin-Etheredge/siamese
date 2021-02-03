import tensorflow as tf
from pathlib import Path
import os
from types import FunctionType


def bool_mask(item, items):
   '''Function for producing/testing a tf bool mask'''
   return item == items


def get_pair(
         item_keys, 
         labels, 
         anchor_key, 
         anchor_label, 
         output_label=None):
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


def create_nway_read_func(items, labels, decode_func, n) :
   #files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(files_dir) / file_glob)))

   #assert tf.size(files_tf) > 0
   filenames = tf.strings.split(items, sep=os.sep)[:, -2:].to_tensor()

   def reader(file_name):
      all_imgs = []
      anchors, others = [], []
      labels_list = []  # Some keras interfaces (like predict) expect a label even when not used, so we'll get them too
      pos_anchor, pos_other, label = get_pair(
         items=items, 
         labels=labels, 
         filenames=filenames, 
         anchor_file_path=file_name, label=1)
      all_imgs.append(decode_func(pos_anchor))
      all_imgs.append(decode_func(pos_other))
      labels_list.append(label)
      labels_list.append(label)
      # TODO will repeat some neagtives, but that's fine
      for _ in range(n-1):
         _, other, label = get_pair(items, labels, filenames, file_name, label=0)
         all_imgs.append(decode_func(other))
         labels_list.append(label)

      return tf.convert_to_tensor(all_imgs), tf.convert_to_tensor(labels_list)

   return reader


def create_n_way_dataset(items, ratio, decode_func,
                         n_way_count, seed=4):
    
   assert n_way_count >= 2, "must be at least 2"

   item_count = int(tf.size(items))
   assert item_count > 0
   assert item_count > n_way_count

   ds = tf.data.Dataset.from_tensor_slices(items)

   ds = ds.shuffle(item_count, seed=seed).take(int(ratio*item_count))

   ds = ds.map(create_nway_read_func(items, decode_func, n=n_way_count),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
   ds = ds.cache()

   return ds


def create_dataset(
      anchor_items: tf.Tensor, 
      anchor_labels: tf.Tensor,
      anchor_decode_func: FunctionType = lambda x: x,
      other_items: tf.Tensor = None, 
      other_labels: tf.Tensor = None,
      batch_size: int = 1, 
      other_decode_func: FunctionType = None,
      repeat=None,
   ):

   if other_items is not None or other_labels is not None:
      assert (other_items is not None and other_labels is not None), "invalid others"
      items = tf.concat((anchor_items, other_items), axis=0)
      labels = tf.concat((anchor_labels, other_labels), axis=0)
   else:
      items = anchor_items
      labels = anchor_labels


   item_count = tf.size(items)
   assert item_count > 0, "No items found"

   label_count = tf.size(labels)
   assert label_count > 0, "No labels found"

   assert label_count == item_count, "Count of labels and items don't match"

   item_ds = tf.data.Dataset.from_tensor_slices(anchor_items)
   label_ds = tf.data.Dataset.from_tensor_slices(labels)
   ds = tf.data.Dataset.zip((item_ds, label_ds))

   # TODO maybe make parier throw out items without 2 elements?
   parier = lambda item, label: get_pair(items, labels, item, label)
   ds = ds.map(parier)

   ds = decoder(ds, anchor_decode_func, other_decode_func)
   
   return ds 

def decoder(
         ds: tf.data.Dataset, 
         anchor_decoder: FunctionType, 
         other_decoder: FunctionType = None) -> tf.data.Dataset:
   if other_decoder is None:
      other_decoder = anchor_decoder
   return ds.map(lambda anchor, other, label: (anchor_decoder(anchor), other_decoder(other), label))

def create_decoder(anchor_decoder: FunctionType,
                   other_decoder: FunctionType = None) -> FunctionType:
   if other_decoder is None:
      other_decoder = anchor_decoder
   
   return lambda anchor, other, label: (anchor_decoder(anchor), other_decoder(other), label)