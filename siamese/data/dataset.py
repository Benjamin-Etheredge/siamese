import tensorflow as tf
from pathlib import Path
import os


def bool_mask(file, files):
   '''Function for producing/testing a tf bool mask'''
   return file == files


def get_pair(
         full_filenames, 
         labels, 
         anchor_file_path, 
         anchor_label, 
         output_label=None):
   '''Get pairwise data'''

   tf.debugging.assert_equal(len(tf.shape(full_filenames)), tf.constant(1))
   tf.debugging.assert_equal(len(tf.shape(labels)), tf.constant(1))

   if output_label is None:
      output_label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)

   anchor_mask = bool_mask(anchor_file_path, full_filenames)

   pos_mask = bool_mask(anchor_label, labels)
   tf.debugging.assert_equal(tf.size(pos_mask), tf.size(labels), f"mask is wrong.\n")
   tf.debugging.assert_less(tf.constant(1), tf.size(labels), f"Split is wrong.\n")
   tf.debugging.assert_equal(len(tf.shape(pos_mask)), tf.constant(1), "mask shape is wrong.\n")

   pos_label_func = lambda: tf.math.logical_xor(pos_mask, anchor_mask)  # XOR prevents anchor file being used
   neg_label_func = lambda: tf.math.logical_not(pos_mask)
   mask = tf.cond(output_label == tf.constant(1), pos_label_func, neg_label_func)
   tf.debugging.assert_equal(len(tf.shape(mask)), tf.constant(1))
   values = tf.boolean_mask(full_filenames, mask)

   tf.debugging.assert_greater(tf.size(values), tf.constant(0), f"Values are empty.\n")
   idx = tf.random.uniform([], 0, tf.size(values), dtype=tf.int32)
   value = tf.gather(values, idx)
   path = value
   sq_path = tf.squeeze(path)
   return anchor_file_path, sq_path, output_label


def create_nway_read_func(files, labels, decode_func, n) :
   #files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(files_dir) / file_glob)))

   #assert tf.size(files_tf) > 0
   filenames = tf.strings.split(files, sep=os.sep)[:, -2:].to_tensor()

   def reader(file_name):
      all_imgs = []
      anchors, others = [], []
      labels_list = []  # Some keras interfaces (like predict) expect a label even when not used, so we'll get them too
      pos_anchor, pos_other, label = get_pair(
         files=files, 
         labels=labels, 
         filenames=filenames, 
         anchor_file_path=file_name, label=1)
      all_imgs.append(decode_func(pos_anchor))
      all_imgs.append(decode_func(pos_other))
      labels_list.append(label)
      labels_list.append(label)
      # TODO will repeat some neagtives, but that's fine
      for _ in range(n-1):
         _, other, label = get_pair(files, labels, filenames, file_name, label=0)
         all_imgs.append(decode_func(other))
         labels_list.append(label)

      return tf.convert_to_tensor(all_imgs), tf.convert_to_tensor(labels_list)

   return reader


def create_n_way_dataset(files, ratio, decode_func,
                         n_way_count, seed=4):
    
   assert n_way_count >= 2, "must be at least 2"

   file_count = int(tf.size(files))
   assert file_count > 0
   assert file_count > n_way_count

   ds = tf.data.Dataset.from_tensor_slices(files)

   ds = ds.shuffle(file_count, seed=seed).take(int(ratio*file_count))

   ds = ds.map(create_nway_read_func(files, decode_func, n=n_way_count),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
   ds = ds.cache()

   return ds


def create_dataset(
      anchor_files, 
      anchor_labels,
      anchor_decode_func,
      other_files=None, 
      other_labels=None,
      batch_size=1, 
      other_decode_func=None,
      shuffle=False,
      repeat=None,
   ):

   if other_files is not None or other_labels is not None:
      assert (other_files is not None and other_labels is not None), "invalid others"
      files = tf.concat((anchor_files, other_files), axis=0)
      labels = tf.concat((anchor_labels, other_labels), axis=0)
   else:
      files = anchor_files
      labels = anchor_labels

   if other_decode_func is None:
      other_decode_func = anchor_decode_func

   file_count = tf.size(files)
   assert file_count > 0, "No files found"

   label_count = tf.size(labels)
   assert label_count > 0, "No labels found"

   assert label_count == file_count, "Count of labels and files don't match"

   file_ds = tf.data.Dataset.from_tensor_slices(anchor_files)
   label_ds = tf.data.Dataset.from_tensor_slices(labels)
   ds = tf.data.Dataset.zip((file_ds, label_ds))

   # TODO maybe make parier throw out items without 2 elements?
   parier = lambda item, label: get_pair(files, labels, item, label)
   ds = ds.map(parier)

   ds = ds.shuffle(buffer_size=int(file_count), reshuffle_each_iteration=True)

   return ds 