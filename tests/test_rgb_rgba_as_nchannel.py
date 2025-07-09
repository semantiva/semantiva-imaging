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
from semantiva_imaging.data_types import (
    RGBImage,
    RGBAImage,
    RGBImageStack,
    RGBAImageStack,
    NChannelImage,
    NChannelImageStack,
)


def test_inheritance_and_channel_info():
    img = RGBImage(np.zeros((2, 2, 3), dtype=np.uint8))
    assert isinstance(img, NChannelImage)
    assert img.channel_info == ("R", "G", "B")

    img4 = RGBAImage(np.zeros((1, 1, 4), dtype=np.float32))
    assert isinstance(img4, NChannelImage)
    assert img4.channel_info == ("R", "G", "B", "A")

    stk = RGBImageStack(np.zeros((2, 2, 2, 3), dtype=np.uint8))
    assert isinstance(stk, NChannelImageStack)
    assert stk.channel_info == ("R", "G", "B")

    stk4 = RGBAImageStack(np.zeros((1, 2, 2, 4), dtype=np.float32))
    assert isinstance(stk4, NChannelImageStack)
    assert stk4.channel_info == ("R", "G", "B", "A")


def test_valid_dtypes():
    for dtype in (np.uint8, np.float32, np.float64):
        assert RGBImage(np.zeros((1, 1, 3), dtype=dtype)).data.dtype == dtype
        assert RGBAImage(np.zeros((1, 1, 4), dtype=dtype)).data.dtype == dtype


def test_autocast_uint16():
    arr3 = np.ones((2, 2, 3), dtype=np.uint16)
    img = RGBImage(arr3)
    assert img.data.dtype == np.float32

    img_no = RGBImage(arr3, auto_cast=False)
    assert img_no.data.dtype == np.uint16

    arr4 = np.ones((1, 2, 2, 4), dtype=np.uint16)
    stk = RGBAImageStack(arr4)
    assert stk.data.dtype == np.float32

    stk_no = RGBAImageStack(arr4, auto_cast=False)
    assert stk_no.data.dtype == np.uint16


def test_shape_mismatch():
    with pytest.raises(AssertionError):
        RGBImage(np.zeros((2, 2, 4), dtype=np.uint8))

    with pytest.raises(AssertionError):
        RGBAImage(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(AssertionError):
        RGBImageStack(np.zeros((1, 2, 2, 4), dtype=np.uint8))

    with pytest.raises(AssertionError):
        RGBAImageStack(np.zeros((1, 2, 2, 3), dtype=np.uint8))
