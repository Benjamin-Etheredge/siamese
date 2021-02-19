from _pytest.fixtures import fixture
import pytest

from ..nway_callback import NWayCallback
from pytest_mock import mocker
from mock import patch, call
from mock import MagicMock
from icecream import ic


class mock_model:
   def __init__(self):
      pass
   def predict_on_batch(*args, **kwargs):
      return True

class mock_ds:
   def __init__(self, size=100):
      self.size=size

   def __iter__(self, *args, **kwargs):
      for _ in range(self, self.size):
         yield True
     


@pytest.mark.parametrize('freq', range(1, 4))
def test_init(freq):
   encoder = MagicMock()
   head = MagicMock()
   ds = MagicMock()
   callback = NWayCallback(encoder=encoder, head=head, nway_ds=ds, freq=freq)
   assert callback.encoder == encoder
   assert callback.head == head
   assert callback.nway_ds == ds
   assert callback.freq == freq


#@pytest.mark.parametrize('count', range(100, 1001, 200))
#@pytest.mark.parametrize('freq', range(1, 101, 20))
@pytest.mark.parametrize('count', [128])
@pytest.mark.parametrize('freq', [7])
def test_on_epoch_end(count, freq):
   encoder = MagicMock()
   encoder.predict_on_batch.return_value = [i for i in range(4)]
   head = MagicMock(return_value=8)
   #head.__call__.return_value = 8
   ds = MagicMock()
   items = list(zip(range(count, 0, -1), range(count, 0, -1)))
   labels = list(range(0, count, 1))
   ds.__iter__.return_value = list(zip(items, labels))

   callback = NWayCallback(encoder=encoder, head=head, nway_ds=ds, freq=freq)
   logs = {}
   for interval in range(1, count+1):
      encoder.reset_mock()
      head.reset_mock()
      ds.reset_mock()
      callback.on_epoch_end(interval, logs=logs)
      if interval % freq == 0:
         ds.__iter__.assert_called_once()
         encoder.assert_has_calls([call.predict_on_batch(item) for item in items])
         for item in items:
            #ic()
            with pytest.raises(AssertionError):
               head.assert_called_with([item, item])
         assert 'nway_acc' in logs
         acc = logs['nway_acc']
         assert 0.0 <= acc <= 1.0
         assert 'nway_avg_dist' in logs
         assert 'nway_avg_var' in logs
         ds.__iter__.assert_called_once()
      else:
         encoder.predict_on_batch.assert_not_called()
         head.assert_not_called()
         ds.__iter__.assert_not_called()
         #head.assert_has_calls(zip(items)
      # TODO test prefix name stuff