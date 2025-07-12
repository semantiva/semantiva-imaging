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

from semantiva_imaging.adapters.opencv_library import (
    SobelEdgeSingleChannelImageProcessor,
)
from semantiva_imaging.data_types import SingleChannelImage, RGBImage
from semantiva_imaging.adapters.opencv_factory import TypeMismatchError


def test_sobel_edge_grayscale_image_processor():
    arr = np.arange(100).reshape(10, 10).astype(np.uint8)
    img = SingleChannelImage(arr)
    proc = SobelEdgeSingleChannelImageProcessor()
    result = proc.process(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    expected = cv2.Sobel(arr, cv2.CV_64F, dx=1, dy=0, ksize=3)
    np.testing.assert_array_equal(result.data, expected)

    result2 = proc.process(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    expected2 = cv2.Sobel(arr, cv2.CV_64F, dx=0, dy=1, ksize=3)
    np.testing.assert_array_equal(result2.data, expected2)


def test_sobel_edge_type_mismatch():
    proc = SobelEdgeSingleChannelImageProcessor()
    bad = RGBImage(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(TypeMismatchError):
        proc.process(bad, dx=1, dy=0, ksize=3)
