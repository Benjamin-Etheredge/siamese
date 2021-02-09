import tensorflow as tf
from types import FunctionType
from .dataset import get_pair
import os


def n_way_read(items: tf.Tensor, labels: tf.Tensor, decode_func: FunctionType, n: int):
    assert tf.size(items) > 0
    assert tf.size(labels) > 0
    assert tf.size(items) == tf.size(labels)

    def foo(item, label):
        all_imgs = []
        anchors, others = [], []
        labels_list = []  # Some keras interfaces (like predict) expect a label even when not used, so we'll get them too
        pos_anchor, pos_other, label = get_pair(
            item_keys=items, 
            labels=labels,
            anchor_key=item,
            anchor_label=label,
            output_label=1)
        all_imgs.append(decode_func(pos_anchor))
        all_imgs.append(decode_func(pos_other))
        labels_list.append(label)
        labels_list.append(label)
        # TODO will repeat some neagtives, but that's fine
        for _ in range(n-2):
            anchor, other, label = get_pair(items, labels, item, label, output_label=0)
            all_imgs.append(decode_func(other))

            labels_list.append(label)


        return tf.convert_to_tensor(all_imgs), tf.convert_to_tensor(labels_list)
    return foo


def create_n_way_dataset(
      items: tf.Tensor, 
      labels: tf.Tensor, 
      ratio: float, 
      anchor_decode_func: FunctionType,
      n_way_count: int):
    """n_way_count is also batch size for ease of use"""
    
    assert n_way_count >= 3, "must be at least 3"
    assert 0.0 <= ratio <= 1.0, "ratio must be between 0 and 1"
    count = int(tf.size(items))
    assert count > 0
    assert count > n_way_count

    item_ds = tf.data.Dataset.from_tensor_slices(items)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((item_ds, label_ds))
    ds = ds.shuffle(count, seed=4).take(int(ratio*count))

    #ds = ds.cache()
    ds_labeled = ds.map(n_way_read(items, labels, anchor_decode_func, n=n_way_count),
                        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(count)

    return ds_labeled
    #return ds_prepared
