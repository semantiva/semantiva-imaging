import numpy as np
import pytest
from semantiva_imaging.data_types import SingleChannelImage, SingleChannelImageStack


def test_uint8_ok():
    img = SingleChannelImage(np.zeros((4, 4), dtype=np.uint8))
    assert img.data.dtype == np.uint8


def test_uint16_auto_cast_true():
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    img = SingleChannelImage(arr)
    assert img.data.dtype == np.float32
    assert np.all(img.data == arr.astype(np.float32))


def test_uint16_auto_cast_false():
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    img = SingleChannelImage(arr, auto_cast=False)
    assert img.data.dtype == np.uint16


def test_shape_error_image():
    with pytest.raises(AssertionError):
        SingleChannelImage(np.zeros((2, 2, 2), dtype=np.uint8))


def test_stack_basic():
    stk = SingleChannelImageStack(np.zeros((3, 5, 5), dtype=np.float32))
    assert stk.data.shape == (3, 5, 5)


def test_stack_shape_error():
    with pytest.raises(AssertionError):
        SingleChannelImageStack(np.zeros((5, 5), dtype=np.uint8))
