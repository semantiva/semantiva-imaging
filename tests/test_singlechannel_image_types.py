# Copyright 2025 Semantiva authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
