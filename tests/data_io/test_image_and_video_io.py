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

"""Integration tests for image and video I/O utilities."""

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

import os
import sys
import numpy as np
import pytest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from semantiva_imaging.data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
    RGBImage,
    RGBAImage,
    RGBAImageStack,
    RGBImageStack,
)
from semantiva_imaging.data_io.loaders_savers import (
    JpgSingleChannelImageLoader,
    JpgSingleChannelImageSaver,
    TiffSingleChannelImageLoader,
    TiffSingleChannelImageSaver,
    JpgRGBImageLoader,
    JpgRGBImageSaver,
    PngRGBImageLoader,
    PngRGBImageSaver,
    TiffRGBImageLoader,
    TiffRGBImageSaver,
    PngRGBAImageLoader,
    PngRGBAImageSaver,
    SingleChannelImageStackVideoLoader,
    SingleChannelImageStackVideoSaver,
    RGBImageStackVideoLoader,
    RGBImageStackVideoSaver,
    AnimatedGifStackLoader,
    AnimatedGifSinglechannelImageStackSaver,
)


@pytest.fixture
def gray_image():
    return SingleChannelImage(np.random.randint(0, 255, (3, 3), dtype=np.uint8))


@pytest.fixture
def rgb_image():
    data = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
    return RGBImage(data)


@pytest.fixture
def rgba_stack():
    data = np.random.randint(0, 255, (2, 8, 8, 4), dtype=np.uint8)
    return RGBAImageStack(data)


@pytest.fixture
def rgb_stack():
    data = np.random.randint(0, 255, (2, 8, 8, 3), dtype=np.uint8)
    return RGBImageStack(data)


@pytest.fixture
def gray_stack():
    data = np.random.randint(0, 255, (2, 8, 8), dtype=np.uint8)
    return SingleChannelImageStack(data)


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def test_single_channel_round_trip_jpg(tmp_dir, gray_image):
    path = os.path.join(tmp_dir, "img.jpg")
    JpgSingleChannelImageSaver().send_data(gray_image, path)
    loaded = JpgSingleChannelImageLoader().get_data(path)
    assert loaded.data.shape == gray_image.data.shape
    assert loaded.data.dtype == np.uint8


def test_single_channel_missing_file():
    loader = JpgSingleChannelImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.get_data("nonexistent.jpg")


def test_rgb_loader_alpha_warning(tmp_dir, caplog):
    rgba_img = RGBAImage(np.random.randint(0, 255, (3, 3, 4), dtype=np.uint8))
    fname = os.path.join(tmp_dir, "alpha.png")
    PngRGBAImageSaver().send_data(rgba_img, fname)
    loader = PngRGBImageLoader()
    with caplog.at_level(logging.WARNING):
        img = loader.get_data(fname)
    assert any("Alpha channel dropped" in r.message for r in caplog.records)
    assert img.data.shape == (3, 3, 3)


def test_rgb_round_trip_video(tmp_dir, rgb_stack):
    path = os.path.join(tmp_dir, "vid.avi")
    RGBImageStackVideoSaver().send_data(rgb_stack, path)
    loaded = RGBImageStackVideoLoader().get_data(path)
    assert loaded.data.shape == rgb_stack.data.shape


def test_single_channel_video_round_trip(tmp_dir, gray_stack):
    path = os.path.join(tmp_dir, "gray.avi")
    SingleChannelImageStackVideoSaver().send_data(gray_stack, path)
    loaded = SingleChannelImageStackVideoLoader().get_data(path)
    assert loaded.data.shape == gray_stack.data.shape


def test_gif_round_trip(tmp_dir, rgba_stack):
    path = os.path.join(tmp_dir, "anim.gif")
    AnimatedGifSinglechannelImageStackSaver().send_data(rgba_stack, path)
    loaded = AnimatedGifStackLoader().get_data(path)
    assert loaded.data.shape == rgba_stack.data.shape
