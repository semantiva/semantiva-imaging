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

"""Domain-specific single-channel data types."""

from typing import Iterator, Sequence, Any, Tuple
import numpy as np
from semantiva.data_types import BaseDataType, DataCollectionType


_ALLOWED_DTYPES = (np.uint8, np.float32, np.float64)


class SingleChannelImage(BaseDataType):
    """
    A class representing a 2D single-channel image.
    This class inherits from `BaseDataType` and is designed to handle 2D single-channel images.
    It validates the input data to ensure it is a 2D NumPy array and supports automatic casting
    of certain data types to `float32` if specified.
    """

    def __init__(self, data: np.ndarray, auto_cast: bool = True, *args, **kwargs):
        """
        Initializes the SingleChannelImage instance.

        Parameters:
            data (numpy.ndarray): The image data to be stored and validated.
            auto_cast (bool): If True, automatically casts uint16 and int16 data types to float32.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If the input data is not of an allowed dtype and auto_cast is True.
        """
        assert data.ndim == 2, f"SingleChannelImage expects 2-D, got ndim={data.ndim}"
        if data.dtype in (np.uint16, np.int16) and auto_cast:
            data = data.astype(np.float32)
        if auto_cast:
            assert (
                data.dtype in _ALLOWED_DTYPES
            ), f"Unsupported dtype {data.dtype}; allowed {_ALLOWED_DTYPES}"
        super().__init__(
            data,
        )

    def validate(self, data: np.ndarray):
        """
        Validates that the input data is a 2D NumPy array.

        Parameters:
            data (numpy.ndarray): The data to validate.

        Raises:
            AssertionError: If the input data is not a NumPy array.
            AssertionError: If the input data is not a 2D array.
        """
        assert isinstance(
            data, np.ndarray
        ), f"Data must be a numpy ndarray, got {type(data)}."
        assert data.ndim == 2, "Data must be a 2D array."
        return data

    def __str__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"

    def __repr__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"


class SingleChannelImageStack(DataCollectionType[SingleChannelImage, np.ndarray]):
    """
    A class representing a 2D image.
    """

    def __init__(self, data: np.ndarray, auto_cast: bool = True, *args, **kwargs):
        assert (
            data.ndim == 3
        ), f"SingleChannelImageStack expects 3-D, got ndim={data.ndim}"
        if data.dtype in (np.uint16, np.int16) and auto_cast:
            data = data.astype(np.float32)
        if auto_cast:
            assert (
                data.dtype in _ALLOWED_DTYPES
            ), f"Unsupported --- dtype {data.dtype}; allowed {_ALLOWED_DTYPES}"
        super().__init__(data, *args, **kwargs)

    def validate(self, data: np.ndarray):
        """
        Validates that the input data is an 3-dimensional NumPy array.

        Parameters:
            data (numpy.ndarray): The data to validate.

        Raises:
            AssertionError: If the input data is not a NumPy array.
        """
        assert isinstance(data, np.ndarray), "Data must be a numpy ndarray."
        assert data.ndim == 3, "Data must be a 3D array (stack of 2D images)"

    def __iter__(self) -> Iterator[SingleChannelImage]:
        """Iterates through the 3D NumPy array, treating each 2D slice as an SingleChannelImage."""
        for i in range(self._data.shape[0]):
            yield SingleChannelImage(self._data[i])

    def append(self, item: SingleChannelImage) -> None:
        """
        Appends a 2D image to the image stack.

        This method takes an `SingleChannelImage` instance and adds its underlying 2D NumPy array
        to the existing 3D NumPy stack. If the stack is empty, it initializes it with the new image.

        Args:
            item (SingleChannelImage): The 2D image to append.

        Raises:
            TypeError: If the item is not an instance of `SingleChannelImage`.
            ValueError: If the image dimensions do not match the existing stack.
        """
        if not isinstance(item, SingleChannelImage):
            raise TypeError(f"Expected SingleChannelImage, got {type(item)}")

        new_image = item.data  # Extract the 2D NumPy array

        if not isinstance(new_image, np.ndarray) or new_image.ndim != 2:
            raise ValueError(f"Expected a 2D NumPy array, got shape {new_image.shape}")

        # If the stack is empty, initialize with the first image
        if self._data.size == 0:
            self._data = np.expand_dims(
                new_image, axis=0
            )  # Convert 2D to 3D with shape (1, H, W)
        else:
            # Ensure the new image has the same dimensions as existing ones
            if new_image.shape != self._data.shape[1:]:
                raise ValueError(
                    f"Image dimensions {new_image.shape} do not match existing stack {self._data.shape[1:]}"
                )

            # Append along the first axis (stack dimension)
            self._data = np.concatenate(
                (self._data, np.expand_dims(new_image, axis=0)), axis=0
            )

    @classmethod
    def _initialize_empty(cls) -> np.ndarray:
        """
        Returns an empty 3D NumPy array for initializing an empty ImageStackDataType.
        """
        return np.empty((0, 0, 0))  # Empty 3D array

    def __len__(self) -> int:
        """
        Returns the number of images in the stack.

        This method returns the number of 2D images stored along the first axis of
        the 3D NumPy array.

        Returns:
            int: The number of images in the stack.
        """
        return self._data.shape[0]

    def __str__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"

    def __repr__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"


class NChannelImage(BaseDataType):
    """Arbitrary-band image (H x W x C).

    Parameters
    ----------
    array : np.ndarray
        Image data with shape ``(H, W, C)``.
    channel_info : Sequence[Any]
        Labels or metadata for each channel (length ``C``). Examples include
        band names, wavelengths, or acquisition parameters.
    auto_cast : bool, default True
        Promote ``uint16``/``int16`` to ``float32`` for OpenCV compatibility.
    """

    def __init__(
        self, array: np.ndarray, channel_info: Sequence[Any], *, auto_cast: bool = True
    ):
        assert array.ndim == 3, f"Expected 3-D array, got ndim={array.ndim}"
        if array.dtype in (np.uint16, np.int16) and auto_cast:
            array = array.astype(np.float32)
        if auto_cast:
            assert (
                array.dtype in _ALLOWED_DTYPES
            ), f"dtype {array.dtype} not allowed; use {_ALLOWED_DTYPES}"
        assert (
            len(channel_info) == array.shape[2]
        ), "channel_info length must match channel count"
        self.channel_info: Tuple[Any, ...] = tuple(channel_info)
        super().__init__(array)

    def validate(self, data: np.ndarray) -> bool:
        """Validate underlying array."""
        assert isinstance(data, np.ndarray), "Data must be a numpy ndarray."
        assert data.ndim == 3, "Data must be a 3D array (H, W, C)."
        return True

    def __str__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"

    def __repr__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"


class NChannelImageStack(BaseDataType):
    """Stack of :class:`NChannelImage` (N x H x W x C)."""

    def __init__(
        self, array: np.ndarray, channel_info: Sequence[Any], *, auto_cast: bool = True
    ):
        assert array.ndim == 4, f"Expected 4-D stack, got ndim={array.ndim}"
        if array.dtype in (np.uint16, np.int16) and auto_cast:
            array = array.astype(np.float32)
        if auto_cast:
            assert (
                array.dtype in _ALLOWED_DTYPES
            ), f"dtype {array.dtype} not allowed; use {_ALLOWED_DTYPES}"
        assert (
            len(channel_info) == array.shape[3]
        ), "channel_info length must match channel count"
        self.channel_info: Tuple[Any, ...] = tuple(channel_info)
        super().__init__(array)

    def validate(self, data: np.ndarray) -> bool:
        """Validate underlying stack array."""
        assert isinstance(data, np.ndarray), "Data must be a numpy ndarray."
        assert data.ndim == 4, "Data must be a 4D array (N, H, W, C)."
        return True

    def __str__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"

    def __repr__(self):
        return f"{self.__class__.__name__}: {self._data.shape}"


class RGBImage(NChannelImage):
    """3-channel image with fixed ("R", "G", "B") labels."""

    def __init__(self, array: np.ndarray, *, auto_cast: bool = True):
        assert (
            array.ndim == 3 and array.shape[2] == 3
        ), "RGBImage expects shape (H, W, 3)"
        super().__init__(array, channel_info=("R", "G", "B"), auto_cast=auto_cast)


class RGBImageStack(NChannelImageStack):
    """Stack of :class:`RGBImage` (N x H x W x 3)."""

    def __init__(self, array: np.ndarray, *, auto_cast: bool = True):
        assert (
            array.ndim == 4 and array.shape[3] == 3
        ), "RGBImageStack expects shape (N, H, W, 3)"
        super().__init__(array, channel_info=("R", "G", "B"), auto_cast=auto_cast)


class RGBAImage(NChannelImage):
    """4-channel image with fixed ("R", "G", "B", "A") labels."""

    def __init__(self, array: np.ndarray, *, auto_cast: bool = True):
        assert (
            array.ndim == 3 and array.shape[2] == 4
        ), "RGBAImage expects shape (H, W, 4)"
        super().__init__(array, channel_info=("R", "G", "B", "A"), auto_cast=auto_cast)


class RGBAImageStack(NChannelImageStack):
    """Stack of :class:`RGBAImage` (N x H x W x 4)."""

    def __init__(self, array: np.ndarray, *, auto_cast: bool = True):
        assert (
            array.ndim == 4 and array.shape[3] == 4
        ), "RGBAImageStack expects shape (N, H, W, 4)"
        super().__init__(array, channel_info=("R", "G", "B", "A"), auto_cast=auto_cast)
