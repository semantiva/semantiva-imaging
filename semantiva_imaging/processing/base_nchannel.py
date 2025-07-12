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

"""Base classes for n-channel image processors and probes."""

from typing import Any
from ..data_types import NChannelImage
from ..processing.processors import DataOperation, DataProbe


def _collect_images(first: NChannelImage, *args, **kwargs) -> list[NChannelImage]:
    """
    Collect all NChannelImage instances from positional and keyword arguments.

    This helper function scans through additional arguments to extract any
    instances of NChannelImage. It ensures that all relevant images are
    included for validation or processing.

    Args:
        first (NChannelImage): The primary image.
        *args: Additional positional arguments, which may include NChannelImage instances.
        **kwargs: Additional keyword arguments, which may include NChannelImage instances.

    Returns:
        list[NChannelImage]: A list of all collected NChannelImage instances.
    """
    images: list[NChannelImage] = [first]
    for arg in args:
        if isinstance(arg, NChannelImage):
            images.append(arg)
    for val in kwargs.values():
        if isinstance(val, NChannelImage):
            images.append(val)
    return images


def _check_nchannel_image_consistency(images: list[NChannelImage]) -> bool:
    """
    Check whether all given NChannelImages have the same shape and channel count.

    Args:
        images (list[NChannelImage]): A list of images to validate.

    Returns:
        bool: True if all images are consistent in shape and number of channels, False otherwise.
    """
    ref_image_shape = images[0].data.shape
    ref_image_channel_count = len(images[0].channel_info)
    for image in images:
        if ref_image_shape != image.data.shape or ref_image_channel_count != len(
            image.channel_info
        ):
            return False
    return True


def _validate_nchannel_inputs(first: NChannelImage, *args, **kwargs) -> bool:
    """
    Validate that all provided NChannelImage instances are consistent.

    Ensures that all NChannelImage inputs (including those passed via args and kwargs)
    have the same shape and number of channels. Raises an error if inconsistency is found.

    Args:
        first (NChannelImage): The primary image.
        *args: Additional positional arguments that may include NChannelImage instances.
        **kwargs: Additional keyword arguments that may include NChannelImage instances.

    Raises:
        TypeError: If the input images differ in shape or channel count.

    Returns:
        bool: True if validation succeeds.
    """
    all_images = _collect_images(first, *args, **kwargs)
    if len(all_images) > 1 and not _check_nchannel_image_consistency(all_images):
        raise TypeError("All input images must have the same shape and channel count.")
    return True


class NChannelImageOperation(DataOperation):
    """
    A DataOperation for processing NChannelImage data.

    """

    def _validate_input(self, input_data, *args, **kwargs) -> bool:
        """
        Validate the input before processing.

        Args:
            input_data (NChannelImage): The primary image to validate.
            *args: Additional inputs.
            **kwargs: Additional inputs.

        Returns:
            bool: True if the validation passes.
        """
        return _validate_nchannel_inputs(input_data, *args, **kwargs)

    def process(self, data: NChannelImage, *args, **kwargs) -> Any:
        """
        Run the operation on the provided NChannelImage.

        Args:
            data (NChannelImage): The input image to process.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            Any: The result of the operation.
        """
        self._validate_input(data, *args, **kwargs)
        return self._process_logic(data, *args, **kwargs)

    @classmethod
    def input_data_type(cls) -> type[NChannelImage]:
        """
        Define the expected input data type.

        Returns:
            type: The expected data type (`NChannelImage`).
        """
        return NChannelImage

    @classmethod
    def output_data_type(cls) -> type[NChannelImage]:
        """
        Define the output data type after processing.

        Returns:
            type: The output data type (`NChannelImage`).
        """
        return NChannelImage


class NChannelImageProbe(DataProbe):
    """
    A DataProbe for analyzing NChannelImage data.

    """

    @classmethod
    def input_data_type(cls) -> type[NChannelImage]:
        """
        Define the expected input data type.

        Returns:
            type: The expected input type (`NChannelImage`).
        """
        return NChannelImage

    def _validate_input(self, input_data, *args, **kwargs) -> bool:
        """
        Validate the input before probing.

        Args:
            input_data (NChannelImage): The primary image to validate.
            *args: Additional inputs.
            **kwargs: Additional inputs.

        Returns:
            bool: True if the validation passes.
        """
        return _validate_nchannel_inputs(input_data, *args, **kwargs)

    def process(self, data: NChannelImage, *args, **kwargs) -> Any:
        """
        Run the probe on the provided NChannelImage.

        Args:
            data (NChannelImage): The input image to analyze.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            Any: The result of the probe .
        """
        self._validate_input(data, *args, **kwargs)
        return self._process_logic(data, *args, **kwargs)
