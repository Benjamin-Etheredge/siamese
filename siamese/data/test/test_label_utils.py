import pytest
import tensorflow as tf
from .. import label_utils as lu

###############################################################################
def test_get_label_from_file_path(): # TODO use os sep
    test_file1 = '/nfs/data/dataset/label/pls.jpg'
    test_file2 = '/nfs/data/dataset/otherlabel/ppls.jpg'
    test_file3 = '/nfs/data/dataset/label/pls2.jpg'
    test_file4 = '/nfs/data/dataset/otherlabel/ppls2.jpg'

    label1 = lu.get_label_from_file_path(test_file1)
    label2 = lu.get_label_from_file_path(test_file2)
    label3 = lu.get_label_from_file_path(test_file3)
    label4 = lu.get_label_from_file_path(test_file4)
    assert label1 == label3 == 'label'
    assert label2 == label4 == 'otherlabel'

###############################################################################

import random
@pytest.fixture(scope="session")
def files_label_in_name(tmpdir_factory, separator='_', label_count=11, count=10):
    data = [(f"label{label_idx}", f"/path/to/data/label{label_idx}_{idx}.file")
              for idx in range(count) 
              for label_idx in range(label_count)]
    random.shuffle(data)
    labels = [item[0] for item in data]
    files = [item[1] for item in data]
    return data, labels, files, separator

def test_get_labels_from_filenames(files_label_in_name):
    _, labels, files, separator = files_label_in_name
    
    retrieved_labels = lu.get_labels_from_filenames(tf.convert_to_tensor(files), separator=separator)
    for y, y_hat in zip(labels, retrieved_labels):
        assert y == y_hat

def test_get_label_from_filename(files_label_in_name):
    _, labels, files, separator = files_label_in_name
    for label, file in zip(labels, files):
        assert label == lu.get_label_from_filename(tf.convert_to_tensor(file), separator=separator)
        assert label == lu.get_label_from_filename(file, separator=separator)
 