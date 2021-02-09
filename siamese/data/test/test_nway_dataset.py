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

   
###############################################################################


#mock_create_nway_read_func = mock.Mock(return_value=lambda x: x)
#create_nway_read_func = mock_create_nway_read_func
#anchor_func_partial = data.create_decode_partial(data.simple_decode, 224, 224)

@pytest.mark.parametrize('n', range(1, 16, 1))
@pytest.mark.parametrize('ratio', np.linspace(-1, 1.1, num=20))
def test_nway_dataset(test_data, n, ratio):
    data_dir, file_paths, items, labels = test_data
    if n < 3 or ratio < 0 or ratio > 1 :
        with pytest.raises(AssertionError):
            ds = create_n_way_dataset(
                items=items,
                labels=labels,
                ratio=ratio,
                anchor_decode_func=lambda x: x,
                n_way_count=n)
    else:
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
        
###############################################################################