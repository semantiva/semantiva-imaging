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

import pytest
import numpy as np
from semantiva_imaging.data_types import NChannelImage
from semantiva_imaging.processing.base_nchannel import (
    NChannelImageOperation,
    NChannelImageProbe,
)


class DummyNChannelImageOperation(NChannelImageOperation):
    """
    A dummy NChannelImageOperation which
    returns the same value. For testing purposes
    """

    def _process_logic(
        self, data: NChannelImage, second_data: NChannelImage
    ) -> NChannelImage:
        return data


class DummyNChannelImageProbe(NChannelImageProbe):
    """
    A dummy NChannelImageProbe which
    returns the same value. For testing purposes
    """

    def _process_logic(self, data: NChannelImage, second_data: NChannelImage) -> int:
        return len(data.channel_info)


@pytest.fixture
def n_channel_image_operation() -> DummyNChannelImageOperation:
    """Returns a DummyNChannelImageOperation instance"""
    return DummyNChannelImageOperation()


@pytest.fixture
def n_channel_image_probe() -> DummyNChannelImageProbe:
    """Returns a DummyNChannelImageProbe instance"""
    return DummyNChannelImageProbe()


@pytest.fixture
def first_nchannel_image() -> NChannelImage:
    """Fixture to generate a RGBA image"""
    random_image = np.random.rand(10, 10, 4)
    return NChannelImage(random_image, channel_info=("R", "G", "B", "A"))


@pytest.fixture
def second_nchannel_image() -> NChannelImage:
    """Fixture to generate a RGB image"""
    random_image = np.random.rand(10, 10, 3)
    return NChannelImage(random_image, channel_info=("R", "G", "B"))


def test_nchannel_operation_good_weather(
    n_channel_image_operation: DummyNChannelImageOperation,
    first_nchannel_image: NChannelImage,
):
    """
    Test if given 2 images with same size and channel count,
    expect NChannelImageOperation to run without errors.
    """

    result_image = n_channel_image_operation(first_nchannel_image, first_nchannel_image)

    assert first_nchannel_image.data.any() == result_image.data.any()
    assert first_nchannel_image.channel_info == result_image.channel_info


def test_nchannel_operation_bad_weather(
    n_channel_image_operation: DummyNChannelImageOperation,
    first_nchannel_image: NChannelImage,
    second_nchannel_image: NChannelImage,
):
    """
    Test if given 2 images with different sizes and channel count,
    expect NChannelImageOperation to raise a TypeError
    """

    with pytest.raises(TypeError):
        n_channel_image_operation(first_nchannel_image, second_nchannel_image)


def test_nchannel_probe_good_weather(
    n_channel_image_probe: DummyNChannelImageProbe,
    first_nchannel_image: NChannelImage,
):
    """
    Test if given 2 images with same size and channel count,
    expect NChannelImageProbe to run without errors.
    """

    result_image = n_channel_image_probe(first_nchannel_image, first_nchannel_image)
    assert len(first_nchannel_image.channel_info) == result_image


def test_nchannel_probe_bad_weather(
    n_channel_image_probe: DummyNChannelImageProbe,
    first_nchannel_image: NChannelImage,
    second_nchannel_image: NChannelImage,
):
    """
    Test if given 2 images with different sizes and channel count,
    expect NChannelImageProbe to raise a TypeError.
    """

    with pytest.raises(TypeError):
        n_channel_image_probe(first_nchannel_image, second_nchannel_image)
