import tensorflow as tf
import os

def get_labels_from_filenames(files: tf.Tensor, separator='_', label_idx=0):
   if not tf.is_tensor(files):
      files = tf.convert_to_tensor(files)

   path_splits = tf.strings.split(files, sep=os.sep)
   filenames = tf.squeeze(path_splits[:, -1:].to_tensor()) # must use slicing for ragged tensor
   filename_splits = tf.strings.split(filenames, sep=separator)
   labels = tf.squeeze(filename_splits[:, :label_idx+1].to_tensor()) # must use slicing for ragged tensor
   return labels


def get_label_from_filename(file: tf.Tensor, separator='_', label_idx=0):
   if not tf.is_tensor(file):
      files = tf.convert_to_tensor(file)

   path_split = tf.strings.split(file, sep=os.sep)
   filename = path_split[-1]
   filename_split = tf.strings.split(filename, sep=separator)
   label = filename_split[label_idx]
   return label


def get_labels_from_files_path(files):
   '''Getter for files in a directory named the label of data'''
   if not tf.is_tensor(files):
      files = tf.convert_to_tensor(files)

   splits = tf.strings.split(files, sep=os.sep)
   labels = tf.squeeze(splits[:, -2:-1].to_tensor())
   return labels


def get_label_from_file_path(file):
   '''Getter for file in a directory named the label of data'''
   if not tf.is_tensor(file):
      files = tf.convert_to_tensor(file)
   all_split = tf.strings.split(file, sep=os.sep)
   labels = all_split[-2]
   return labels
