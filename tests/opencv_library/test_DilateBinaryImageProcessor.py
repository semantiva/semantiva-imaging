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
import cv2
import pytest

from semantiva_imaging.adapters.opencv_library import DilateSingleChannelImageProcessor
from semantiva_imaging.data_types import SingleChannelImage, RGBImage
from semantiva_imaging.adapters.opencv_factory import TypeMismatchError


def test_dilate_binary_image_processor():
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[4:6, 4:6] = 1
    img = SingleChannelImage(arr)
    kernel = np.ones((3, 3), dtype=np.uint8)
    proc = DilateSingleChannelImageProcessor()
    result = proc.process(img, kernel=kernel, iterations=1)
    expected = cv2.dilate(arr, kernel=kernel, iterations=1)
    np.testing.assert_array_equal(result.data, expected)

    result2 = proc.process(img, kernel=kernel, iterations=2)
    expected2 = cv2.dilate(arr, kernel=kernel, iterations=2)
    np.testing.assert_array_equal(result2.data, expected2)


def test_dilate_type_mismatch():
    proc = DilateSingleChannelImageProcessor()
    bad = RGBImage(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(TypeMismatchError):
        proc.process(bad, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
