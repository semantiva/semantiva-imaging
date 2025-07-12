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

"""Concrete image operations used in processing pipelines."""

import numpy as np
from ..data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
)
from ..processing.processors import (
    SingleChannelImageOperation,
    SingleChannelImageStackToImageProjector,
)


class ImageSubtraction(SingleChannelImageOperation):
    """
    Substracts one image from another.
    """

    def _process_logic(
        self, data: SingleChannelImage, image_to_subtract: SingleChannelImage
    ) -> SingleChannelImage:
        """
        Subtracts one image from another.

        Parameters:
            data (SingleChannelImage): The original image data.
            image_to_subtract (SingleChannelImage): The image data to subtract.

        Returns:
            SingleChannelImage: The result of the subtraction operation.
        """
        return SingleChannelImage(np.subtract(data.data, image_to_subtract.data))


class ImageAddition(SingleChannelImageOperation):
    """
    Adds two images together.
    """

    def _process_logic(
        self, data: SingleChannelImage, image_to_add: SingleChannelImage
    ) -> SingleChannelImage:
        """
        Adds one image to another.

        Parameters:
            data (SingleChannelImage): The original image data.
            image_to_add (SingleChannelImage): The image data to add.

        Returns:
            SingleChannelImage: The result of the addition operation.
        """
        return SingleChannelImage(np.add(data.data, image_to_add.data))


class ImageCropper(SingleChannelImageOperation):
    """
    Crops a rectangular region from the input image.
    """

    def _process_logic(
        self,
        data: SingleChannelImage,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> SingleChannelImage:
        """
        Crop a rectangular region from the input image.

        Parameters:
            data (SingleChannelImage): The original image data.
            x_start (int): The starting x-coordinate of the cropped region.
            x_end (int): The ending x-coordinate of the cropped region.
            y_start (int): The starting y-coordinate of the cropped region.
            y_end (int): The ending y-coordinate of the cropped region.

        Returns:
            SingleChannelImage: The cropped region of the image.

        Raises:
            ValueError: If the specified cropped region is out of bounds.
        """

        # Ensure the region is within bounds
        if not 0 <= x_start < x_end <= data.data.shape[1]:
            raise ValueError(
                f"x-coordinates out of bounds: x_start={x_start}, x_end={x_end}, width={data.data.shape[1]}"
            )
        if not 0 <= y_start < y_end <= data.data.shape[0]:
            raise ValueError(
                f"y-coordinates out of bounds: y_start={y_start}, y_end={y_end}, height={data.data.shape[0]}"
            )

        cropped_array = data.data[y_start:y_end, x_start:x_end]
        return SingleChannelImage(cropped_array)


class StackToImageMeanProjector(SingleChannelImageStackToImageProjector):
    """
    Projects a stack of images into a single image by computing the mean pixel value among the slices.
    """

    def _process_logic(self, data: SingleChannelImageStack) -> SingleChannelImage:
        """
        Computes the mean projection of an image stack.

        Parameters:
            data (SingleChannelImageStack): The input image stack, represented as a 3D NumPy array
                                        (stack of 2D images).

        Returns:
            SingleChannelImage: A single 2D image resulting from the mean projection of the stack.

        Raises:
            ValueError: If the input data is not a 3D NumPy array.

        """
        # Compute the mean along the stack (first axis)
        return SingleChannelImage(np.mean(data.data, axis=0))


class ImageNormalizerOperation(SingleChannelImageOperation):
    """
    Linear normalization of an image data to a specified range.

    """

    def _process_logic(
        self,
        data: SingleChannelImage,
        min_value: float,
        max_value: float,
        *args,
        **kwargs,
    ) -> SingleChannelImage:
        """
        Perform normalization on the image data to scale its pixel values
        linearly to the specified range `[min_value, max_value]`.

        Parameters:
            data (SingleChannelImage): The image data to normalize.
            min_value (float): The minimum value of the target range.
            max_value (float): The maximum value of the target range.
            *args, **kwargs: Additional parameters for compatibility.

        Returns:
            SingleChannelImage: The normalized image data with values in `[min_value, max_value]`.
        """

        image_array = data.data
        arr_min = np.min(image_array)
        arr_max = np.max(image_array)

        # Avoid division by zero in case all values in the array are the same
        if arr_max - arr_min == 0:
            return SingleChannelImage(
                np.full_like(image_array, (min_value + max_value) / 2)
            )

        # Linear scaling
        scaled_arr = min_value + (image_array - arr_min) * (max_value - min_value) / (
            arr_max - arr_min
        )
        return SingleChannelImage(scaled_arr)


class SingleChannelImageStackSideBySideProjector(
    SingleChannelImageStackToImageProjector
):
    """
    Projects a stack of images into a single image by concatenating the images side by side.
    """

    def _process_logic(self, data: SingleChannelImageStack) -> SingleChannelImage:
        """
        Concatenates the images in the stack side by side.

        Parameters:
            data (SingleChannelImageStack): The input image stack, represented as a 3D NumPy array
                                        (stack of 2D images).

        Returns:
            SingleChannelImage: A single 2D image resulting from the concatenation of images in the stack.

        Raises:
            ValueError: If the input data is not a 3D NumPy array.

        """
        # Concatenate the images along the horizontal axis (axis=1)
        return SingleChannelImage((np.hstack(tuple(data.data))))
