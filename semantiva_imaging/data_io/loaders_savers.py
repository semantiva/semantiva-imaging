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

"""Loaders and savers for image data."""

from typing import Any, Dict, Tuple, List
from PIL import Image
import numpy as np
from semantiva.context_processors.context_types import ContextType
from .io import (
    SingleChannelImageDataSource,
    SingleChannelImageStackSource,
    SingleChannelImageDataSink,
    SingleChannelImageStackSink,
    SingleChannelImageStackPayloadSource,
)
from ..data_types import SingleChannelImage, SingleChannelImageStack


class NpzSingleChannelImageLoader(SingleChannelImageDataSource):
    """
    Concrete implementation of ImageDataTypeSource for loading image data from .npz files.

    This class provides functionality to load a single array from a `.npz` file
    as a :class:`SingleChannelImage`.
    """

    @classmethod
    def _get_data(cls, path: str, *args, **kwargs):
        """
        Loads the single array from a .npz file and returns it as a ``SingleChannelImage``.

        Assumes the `.npz` file contains only one array.

        Parameters:
            path (str): The path to the .npz file containing the image data.

        Returns:
            SingleChannelImage: The loaded image data.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the .npz file does not contain exactly one array.
            ValueError: If the array is not a 2D array.
        """
        try:
            # Load the .npz file
            with np.load(path) as data:
                # Validate the file contains exactly one array
                if len(data.files) != 1:
                    raise ValueError(
                        f"The file {path} must contain exactly one array, "
                        f"but found {len(data.files)}."
                    )

                # Get the array
                array_name = data.files[0]
                array = data[array_name]

                # Validate the array shape
                if array.ndim != 2:
                    raise ValueError(f"The array in {path} is not a 2D array.")

                # Wrap the array in a SingleChannelImage object
                return SingleChannelImage(array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading image data from {path}: {e}") from e


class NpzImageDataSaver(SingleChannelImageDataSink):
    """Save a ``SingleChannelImage`` to an ``.npz`` file."""

    def _send_data(self, data: SingleChannelImage, path: str):
        """
        Saves the ``SingleChannelImage`` as a `.npz` file at the specified path.

        Parameters:
            data (SingleChannelImage): The image data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImage``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")

        try:
            # Save the data to an .npz file
            np.savez(path, image=data.data)
        except Exception as e:
            raise IOError(f"Error saving SingleChannelImage to {path}: {e}") from e


class NpzSingleChannelImageStackDataLoader(SingleChannelImageStackSource):
    """
    Concrete implementation of ImageStackSource for loading image stacks from .npz files.

    This class provides functionality to load a single 3D array from a `.npz` file
    as a :class:`SingleChannelImageStack`.
    """

    @classmethod
    def _get_data(cls, path: str):
        """
        Loads the single 3D array from a .npz file and returns it as a ``SingleChannelImageStack``.

        Assumes the `.npz` file contains only one array, which is 3D.

        Parameters:
            path (str): The path to the .npz file containing the image stack data.

        Returns:
            SingleChannelImageStack: The loaded image stack data.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the .npz file does not contain exactly one array.
            ValueError: If the array is not a 3D array.
        """
        try:
            # Load the .npz file
            with np.load(path) as data:
                # Validate the file contains exactly one array
                if len(data.files) != 1:
                    raise ValueError(
                        f"The file {path} must contain exactly one array, "
                        f"but found {len(data.files)}."
                    )

                # Get the array
                array_name = data.files[0]
                array = data[array_name]

                # Validate that it a 3D array
                if array.ndim != 3:
                    raise ValueError(f"The array in {path} is not a 3D array.")

                # Wrap the array in a SingleChannelImageStack object
                return SingleChannelImageStack(array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading image stack data from {path}: {e}") from e


class NpzImageStackDataSaver(SingleChannelImageStackSink):
    """Save a ``SingleChannelImageStack`` to an ``.npz`` file."""

    def _send_data(self, data: SingleChannelImageStack, path: str):
        """
        Saves the ``SingleChannelImageStack`` as a `.npz` file at the specified path.

        Parameters:
            data (SingleChannelImageStack): The image stack data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImageStack``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )

        try:
            # Save the data to an .npz file
            np.savez(path, image_stack=data.data)
        except Exception as e:
            raise IOError(f"Error saving SingleChannelImageStack to {path}: {e}") from e


class PngImageLoader(SingleChannelImageDataSource):
    """
    Concrete implementation of ImageDataTypeSource for loading image data from PNG files.

    This class provides functionality to load a PNG image as a :class:`SingleChannelImage`.
    """

    @classmethod
    def _get_data(cls, path: str):
        """
        Loads a PNG image from the specified file path and returns it as a ``SingleChannelImage``.

        Parameters:
            path (str): The path to the PNG file.

        Returns:
            SingleChannelImage: The loaded image data.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the file cannot be opened or does not contain valid image data.
        """
        try:
            # Open the PNG image
            with Image.open(path) as img:
                # Convert the image to single-channel and load it as a NumPy array
                image_array = np.asarray(img.convert("L"))
                # Wrap the array in a SingleChannelImage object
                return SingleChannelImage(image_array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading PNG image from {path}: {e}") from e


class PngImageSaver(SingleChannelImageDataSink):
    """
    Concrete implementation of ImageDataTypeSink for saving image data to PNG files.

    This class provides functionality to save a ``SingleChannelImage`` object as a PNG image.
    """

    def _send_data(self, data: SingleChannelImage, path: str):
        """
        Saves the ``SingleChannelImage`` as a PNG file at the specified path.

        Parameters:
            data (SingleChannelImage): The image data to be saved.
            path (str): The file path to save the PNG image.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImage``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")

        try:
            # Convert the NumPy array to a PIL image
            img = Image.fromarray(data.data.astype(np.uint8))
            # Save the image as a PNG
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG image to {path}: {e}") from e


class PNGImageStackSaver(SingleChannelImageStackSink):
    """
    Concrete implementation of ImageDataSink for saving multi-frame image data (``SingleChannelImageStack``)
    as sequentially numbered PNG files.

    Each frame in the ``SingleChannelImageStack`` is saved as a separate PNG
    file with filenames numbered sequentially (e.g., ``frame_000.png``).
    """

    def _send_data(self, data: SingleChannelImageStack, base_path: str):
        """
        Saves the ``SingleChannelImageStack`` as sequentially numbered PNG files.

        Parameters:
            data (SingleChannelImageStack): The image stack data to be saved.
            base_path (str): The base file path to save PNG files. A number will
                             be appended to this path for each frame.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImageStack``.
            IOError: If any frame cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )

        try:
            # Iterate through each frame in the stack
            for i, frame in enumerate(data.data):
                # Convert the frame to a PIL image
                img = Image.fromarray(frame.astype(np.uint8))
                # Generate a filename with sequential numbering
                file_path = f"{base_path}_{i:03d}.png"
                # Save the image as a PNG
                img.save(file_path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG image stack: {e}") from e


class ImageDataRandomGenerator(SingleChannelImageDataSource):
    """
    A random generator for creating ``SingleChannelImage`` objects with random data.
    """

    @classmethod
    def _get_data(cls, shape: tuple[int, int]):
        """
        Generates a dummy ``SingleChannelImage`` with random values.

        Parameters:
            shape (tuple[int, int]): The shape (rows, columns) of the generated image data.

        Returns:
            SingleChannelImage: A dummy image data object containing a 2D array of random values.

        Raises:
            ValueError: If the provided shape does not have exactly two dimensions.
        """

        # Validate that the shape represents a 2D array
        if len(shape) != 2:
            raise ValueError(
                f"Shape must be a tuple with two dimensions, but got {shape}."
            )
        return SingleChannelImage(np.random.rand(*shape))


class TwoDGaussianImageGenerator(SingleChannelImageDataSource):
    """Generates an image with a 2D Gaussian signal with optional rotation."""

    @classmethod
    def _get_data(
        self,
        x_0: float | int,  # x position
        y_0: float | int,  # y position
        std_dev: float | tuple[float, float],  # Allow single float or tuple
        amplitude: float,
        angle: float = 0.0,  # Rotation angle in degrees
        image_size: tuple[int, int] = (1024, 1024),  # Default image size
    ):
        """
        Generates a 2D Gaussian image centered at a given position with an optional rotation.

        Parameters:
            center (tuple[int, int]): (x, y) coordinates of the Gaussian center.
            std_dev (float | tuple[float, float]): Standard deviation (sigma_x, sigma_y).
                - If a single float is provided, both directions use the same value.
            amplitude (float): Amplitude (peak intensity) of the Gaussian.
            angle (float): Rotation angle in degrees.
            image_size (tuple[int, int]): The shape (height, width) of the generated image (default: (1024, 1024)).

        Returns:
            SingleChannelImage: A n image with a 2D Gaussian shape.
        """
        x_center, y_center = x_0, y_0

        if isinstance(std_dev, (int, float)):
            std_dev_x, std_dev_y = std_dev, std_dev
        else:
            std_dev_x, std_dev_y = std_dev

        if len(image_size) != 2:
            raise ValueError(
                f"image_size must be a tuple of two dimensions, but got {image_size}."
            )

        # Generate the Gaussian grid
        x = np.arange(image_size[1])  # width (columns)
        y = np.arange(image_size[0])  # height (rows)
        x_grid, y_grid = np.meshgrid(x, y)

        # Compute the Gaussian function
        x_shifted = x_grid - x_center
        y_shifted = y_grid - y_center

        theta = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_rotated = cos_theta * x_shifted - sin_theta * y_shifted
        y_rotated = sin_theta * x_shifted + cos_theta * y_shifted

        z = amplitude * np.exp(
            -((x_rotated**2) / (2 * std_dev_x**2) + (y_rotated**2) / (2 * std_dev_y**2))
        )

        return SingleChannelImage(z)


class ParametricImageStackGenerator(SingleChannelImageStackSource):
    def __init__(
        self,
        num_frames: int,
        parametric_expressions: Dict[str, str],
        param_ranges: Dict[str, Tuple[float, float]],
        image_generator: Any,
        image_generator_params: Dict[str, Any],
    ):
        """
        Creates an image stack of signals where parameters evolve according to parametric expressions.

        Parameters:
            num_frames (int): Number of images in the stack.
            parametric_expressions (Dict[str, str]): Dictionary specifying how each parameter evolves.
                Keys correspond to parameter names used in the image generator.
                Values are string expressions that evaluate t -> value.
            param_ranges (Dict[str, Tuple[float, float]]): Dictionary specifying parameter ranges for time t.
            image_generator (Any): An image generator.
            image_generator_params (Dict[str, Any]): Dictionary of additional static parameters for the image generator.
        """
        self.num_frames = num_frames
        self.parametric_expressions = {
            key: eval(f"lambda t: {expr}") if isinstance(expr, str) else expr
            for key, expr in parametric_expressions.items()
        }
        self.param_ranges = param_ranges
        self.image_generator = image_generator
        self.image_generator_params = image_generator_params

    def _evaluate_param(self, param_name: str, t: float) -> float:
        """
        Evaluates the given parameter function.

        Parameters:
            param_name (str): The name of the parameter to evaluate.
            t (float): The time value at which to evaluate the parameter.

        Returns:
            float: The evaluated parameter value.
        """
        return self.parametric_expressions[param_name](t)

    def _get_data(cls):
        """
        Generates a stack of images with evolving parameters.

        Returns:
            SingleChannelImageStack: The generated image stack data.
        """
        images = [
            cls.image_generator.get_data(
                **{
                    key: cls._evaluate_param(key, t)
                    for key in cls.parametric_expressions
                },
                **cls.image_generator_params,
            )
            for t in cls.t_values
        ]
        return SingleChannelImageStack(
            images
            if isinstance(images, np.ndarray)
            else np.stack([img.data for img in images])
        )

    @property
    def t_values(self):
        """Range of ``t`` values used for the parametric expressions."""
        t_values = np.linspace(
            self.param_ranges["t"][0], self.param_ranges["t"][1], self.num_frames
        )

        return t_values


class SingleChannelImageStackRandomGenerator(SingleChannelImageStackSource):
    """
    A random generator for creating ``SingleChannelImageStack`` objects with random data.
    """

    @classmethod
    def _get_data(cls, shape: tuple[int, int, int]):
        """
        Generates a dummy ``SingleChannelImageStack`` with random values.

        Parameters:
            shape (tuple[int, int, int]): The shape (slices, rows, columns) of the generated
                                          image stack data.

        Returns:
            SingleChannelImageStack: A dummy image stack data object containing a 3D array of random values.

        Raises:
            ValueError: If the provided shape does not have exactly three dimensions.
        """
        # Validate that the shape represents a 3D array
        if len(shape) != 3:
            raise ValueError(
                f"Shape must be a tuple with three dimensions, but got {shape}."
            )

        return SingleChannelImageStack(np.random.rand(*shape))


class SingleChannelImageStackPayloadRandomGenerator(
    SingleChannelImageStackPayloadSource
):
    """
    A random generator for producing payloads containing ``SingleChannelImageStack`` and ``ContextType``.
    """

    @classmethod
    def _injected_context_keys(cls) -> List[str]:
        """
        Returns a list of context keys that are injected into the payload.

        This method can be overridden by subclasses to specify additional context keys.

        Returns:
            List[str]: A list of context keys.
        """
        return ["image_stack_payload"]

    def _get_payload(
        self, *args, **kwargs
    ) -> tuple[SingleChannelImageStack, ContextType]:
        """
        Generates and returns a dummy payload.

        The payload consists of:
        - ``SingleChannelImageStack``: A 3D NumPy array with random data (a stack of 2D images).
        - `ContextType`: A dictionary containing dummy contextual information.

        Parameters:
            *args: Additional arguments for customization (not used in this implementation).
            **kwargs: Additional keyword arguments for customization (not used in this implementation).

        Returns:
            tuple[SingleChannelImageStack, dict]:
                A tuple containing the ``SingleChannelImageStack`` with dummy data and a
                ``ContextType`` dictionary with dummy metadata.
        """
        # Generate a dummy 3D NumPy array (stack of 10 images, each 256x256)
        dummy_stack = np.random.rand(10, 256, 256)  # Example stack of 10 images

        # Wrap the stack in an SingleChannelImageStack and return the payload
        return SingleChannelImageStack(dummy_stack), ContextType()
