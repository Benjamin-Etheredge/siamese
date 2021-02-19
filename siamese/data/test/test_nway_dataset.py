from ..nway_dataset import create_n_way_dataset, n_way_read
from unittest import mock
import pytest
import tensorflow as tf
from icecream import ic
import numpy as np

###############################################################################
# Fixtures
###############################################################################

@pytest.fixture(scope="session")
def test_data(tmpdir_factory, count=1000, label_count=13):
    fn = tmpdir_factory.mktemp("data")
    labels = []
    file_paths = []
    filenames = []
    for idx in range(count):
        filename = f"item_{idx}.file"
        filenames.append(filename)

        path = str(fn.join(filename))

        file_paths.append(path)
        labels.append(idx % label_count)

    return fn, *[tf.convert_to_tensor(item) for item in (file_paths, filenames, labels)]

   
import contextlib

@contextlib.contextmanager
def dummy_checker():
    yield None

###############################################################################

from ..nway_dataset import n_way_read
@pytest.mark.parametrize('n', range(0, 16, 1))
def test_nway_read(test_data, n):
    data_dir, file_paths, items, labels = test_data
    with pytest.raises(AssertionError) if n < 3 else dummy_checker():
        read_func = n_way_read(items, labels, lambda x: x, n)
        for item, label in zip(items, labels):
            batch = list(iter(read_func(item, label)))
            assert len(batch[0]) == n
            batch_items = batch[0]
            batch_labels = batch[1]
            assert len(batch_items) == len(batch_labels)
            assert batch_labels[0] == batch_labels[1]
            for sub_label in batch_labels[2:]:
                assert sub_label != batch_labels[0]


#mock_create_nway_read_func = mock.Mock(return_value=lambda x: x)
#create_nway_read_func = mock_create_nway_read_func
#anchor_func_partial = data.create_decode_partial(data.simple_decode, 224, 224)

# TODO mock n_way_read or paramaterize it for injecting
@pytest.mark.parametrize('n', [1, 2, 3, 4, 17])
@pytest.mark.parametrize('ratio', np.linspace(-1, 1.1, num=10))
def test_nway_dataset(test_data, n, ratio):
    data_dir, file_paths, items, labels = test_data
    with pytest.raises(AssertionError) if (n < 3 or ratio < 0 or ratio > 1) \
            else dummy_checker():
        ds = create_n_way_dataset(
                items=items,
                labels=labels,
                ratio=ratio,
                anchor_decode_func=lambda x: x,
                n_way_count=n)
        for batch_items, batch_labels in ds:
            assert len(batch_items) == len(batch_labels) == n
            assert batch_labels[0] == batch_labels[1]
            for idx in range(3, len(batch_items)):
                assert batch_labels[idx-1] == batch_labels[idx]

        nway_items_count = len(list(iter(ds)))
        assert nway_items_count == int(len(items) * ratio)

        # Make sure items are the same each iteration
        items_1 = [batch_items for batch_items, _ in ds]
        items_2 = [batch_items for batch_items, _ in ds]
        for batch_1, batch_2 in zip(items_1, items_2):
            for item1, item2 in zip(batch_1, batch_2):
                    assert item1 == item2
        
###############################################################################