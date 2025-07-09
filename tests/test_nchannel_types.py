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
from semantiva_imaging.data_types import NChannelImage, NChannelImageStack


def test_valid_images_and_stacks():
    img1 = NChannelImage(np.zeros((4, 4, 1), dtype=np.uint8), ["only"])
    assert img1.data.shape == (4, 4, 1)

    img3 = NChannelImage(np.zeros((2, 3, 3), dtype=np.float32), ["r", "g", "b"])
    assert img3.data.dtype == np.float32

    img7 = NChannelImage(np.zeros((1, 1, 7), dtype=np.float64), list(range(7)))
    assert len(img7.channel_info) == 7

    stack = NChannelImageStack(np.zeros((5, 2, 2, 3), dtype=np.uint8), ["r", "g", "b"])
    assert stack.data.shape == (5, 2, 2, 3)


def test_mismatched_channel_info():
    with pytest.raises(AssertionError):
        NChannelImage(np.zeros((2, 2, 2), dtype=np.uint8), ["a"])

    with pytest.raises(AssertionError):
        NChannelImageStack(np.zeros((1, 2, 2, 3), dtype=np.uint8), [1, 2])


def test_wrong_ndim():
    with pytest.raises(AssertionError):
        NChannelImage(np.zeros((2, 2), dtype=np.uint8), [])

    with pytest.raises(AssertionError):
        NChannelImageStack(np.zeros((2, 2, 3), dtype=np.uint8), [1, 2, 3])


def test_uint16_autocast():
    arr = np.ones((2, 2, 1), dtype=np.uint16)
    img = NChannelImage(arr, ["a"])
    assert img.data.dtype == np.float32

    img_no = NChannelImage(arr, ["a"], auto_cast=False)
    assert img_no.data.dtype == np.uint16

    stack = NChannelImageStack(arr[np.newaxis, ...], ["a"])
    assert stack.data.dtype == np.float32

    stack_no = NChannelImageStack(arr[np.newaxis, ...], ["a"], auto_cast=False)
    assert stack_no.data.dtype == np.uint16
