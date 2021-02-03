from .. import dataset
from ..dataset import create_dataset, create_nway_read_func
from unittest import mock
import pytest
import tensorflow as tf

###############################################################################
# Fixtures
###############################################################################

@pytest.fixture(scope="session")
def test_data(tmpdir_factory, count=10):
    fn = tmpdir_factory.mktemp("data")
    labels = []
    file_paths = []
    filenames = []
    for idx in range(count):
        filename = f"item_{idx}.file"
        filenames.append(filename)

        path = str(fn.join(filename))

        file_paths.append(path)
        labels.append(idx % 3)

    return fn, *[tf.convert_to_tensor(item) for item in (file_paths, filenames, labels)]

   
###############################################################################

def test_bool_mask_file(test_data):
    _, file_paths, _, _ = test_data
    print(type(file_paths))
    for file_path in file_paths:
        mask = dataset.bool_mask(file_path, file_paths)
        assert tf.size(mask) == tf.size(file_paths)
        num_hot =  tf.reduce_sum(tf.cast(mask, tf.int32), keepdims=False)
        assert num_hot == tf.constant(1), \
            f"num_hot ({num_hot}) not equal to 1 - file: {file_path}\nfiles: {file_paths}"

def count_hot(mask):
    return tf.reduce_sum(tf.cast(mask, tf.int32), keepdims=False)


from collections import Counter
def test_bool_mask_label(test_data):
    data_dir, file_paths, filenames, labels = test_data
    label_counts = Counter([str(label) for label in labels])

    for label in labels:
        mask = dataset.bool_mask(label, labels)
        assert tf.size(mask) == tf.size(labels)
        num_hot = count_hot(mask)
        assert num_hot == tf.constant(label_counts[str(label)]), \
            f"num_hot ({num_hot}) not equal to {label_counts[label]}"



#all_files_tf = tf.io.gfile.glob(data_dir + '/*/*.jpg')
#all_labels = data.get_labels_from_files_path(all_files_tf)
#@pytest.mark.parametrize('img_file', files)
def test_get_pair(test_data):
    # TODO test different input shapes
    data_dir, file_paths, filenames, labels = test_data
    assert len(labels) > 0

    for file_path, anchor_label in zip(file_paths, labels):
        # test labels given and not given
        for input_label in [None, 0, 1]:
            for _ in range(10):
                anchor_file, other_file, label = dataset.get_pair(
                    file_paths, labels, file_path, anchor_label,
                    output_label=input_label)
                #print(other_file)
                other_file_idx = tf.squeeze(tf.where(other_file == file_paths))
                other_label = labels[other_file_idx]
                assert anchor_file != other_file, \
                    f"got a duplicate file - label: {label} - anchor_label: {anchor_label} - other_label: {other_label}"
                if label == 1:
                    assert anchor_label == other_label
                else:
                    assert anchor_label != other_label
                if input_label:
                    assert input_label == label


#mock_create_nway_read_func = mock.Mock(return_value=lambda x: x)
#create_nway_read_func = mock_create_nway_read_func
#anchor_func_partial = data.create_decode_partial(data.simple_decode, 224, 224)
'''
def test_nway_dataset():
    ds = data.create_n_way_dataset(
        data_directory_name=data_dir,
        batch_size=4,
        anchor_decode_func=anchor_func_partial,
        n_way_count=4)
    for item, label in ds:
        print(len(item[0]))
        print(len(item[1]))
        print(label)
        break
'''
###############################################################################

def test_create_dataset(test_data):
    data_dir, file_paths, filenames, labels = test_data
    ds = create_dataset(
        anchor_items=file_paths,
        anchor_labels=labels,
        anchor_decode_func=lambda x: x,
    )
    for anchor, other, label in ds:
        assert label == 0 or label == 1, "Incorrect label value"
        assert anchor != other, "Anchor repeated"

        anchor_idx = [idx for idx, file in enumerate(file_paths) if file == anchor]
        other_idx = [idx for idx, file in enumerate(file_paths) if file == other]
        assert len(anchor_idx) == 1 == len(other_idx), "repeated file"

        anchor_label = labels[anchor_idx[0]]
        other_label = labels[other_idx[0]]
        do_labels_match = (anchor_label == other_label) == (label==1) 
        assert do_labels_match, "incorrect label"

from ..dataset import create_decoder
def test_create_decoder(test_data):
    func = create_decoder(lambda x: x*2, lambda x: x*4) 
    assert func(1, 2, 4) == ((2, 8, 4))

    data_dir, file_paths, filenames, labels = test_data
    ds = create_dataset(
        anchor_items=file_paths,
        anchor_labels=labels,
        anchor_decode_func=lambda x: x,
    )
    #ds.map(func)
    # TODO 
