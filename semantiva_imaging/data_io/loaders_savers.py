from PIL import Image
import numpy as np
from typing import Any, Dict, Tuple, List
from semantiva.context_processors.context_types import ContextType
from .io import (
    ImageDataSource,
    ImageStackSource,
    ImageDataSink,
    ImageStackDataSink,
    ImageStackPayloadSource,
)
from ..data_types import ImageDataType, ImageStackDataType


class NpzImageDataTypeLoader(ImageDataSource):
    """
    Concrete implementation of ImageDataTypeSource for loading image data from .npz files.

    This class provides functionality to load a single array from a `.npz` file
    as an `ImageDataType`.
    """

    def _get_data(self, path: str) -> ImageDataType:
        """
        Loads the single array from a .npz file and returns it as an `ImageDataType`.

        Assumes the `.npz` file contains only one array.

        Parameters:
            path (str): The path to the .npz file containing the image data.

        Returns:
            ImageDataType: The loaded image data.

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
                        f"The file {path} must contain exactly one array, but found {len(data.files)}."
                    )

                # Get the array
                array_name = data.files[0]
                array = data[array_name]

                # Validate the array shape
                if array.ndim != 2:
                    raise ValueError(f"The array in {path} is not a 2D array.")

                # Wrap the array in an ImageDataType object
                return ImageDataType(array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading image data from {path}: {e}") from e


class NpzImageDataSaver(ImageDataSink):
    """
    Concrete implementation of ImageDataTypeSink for saving `ImageDataType` objects to .npz files.

    This class provides functionality to save an `ImageDataType` object into a `.npz` file.
    """

    def _send_data(self, data: ImageDataType, path: str):
        """
        Saves the `ImageDataType` as a `.npz` file at the specified path.

        Parameters:
            data (ImageDataType): The image data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not an `ImageDataType`.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of ImageDataType.")

        try:
            # Save the data to an .npz file
            np.savez(path, image=data.data)
        except Exception as e:
            raise IOError(f"Error saving ImageDataType to {path}: {e}") from e


class NpzImageStackDataLoader(ImageStackSource):
    """
    Concrete implementation of ImageStackDataTypeSource for loading image stack data from .npz files.

    This class provides functionality to load a single 3D array from a `.npz` file
    as an `ImageStackDataType`.
    """

    def _get_data(self, path: str) -> ImageStackDataType:
        """
        Loads the single 3D array from a .npz file and returns it as an `ImageStackDataType`.

        Assumes the `.npz` file contains only one array, which is 3D.

        Parameters:
            path (str): The path to the .npz file containing the image stack data.

        Returns:
            ImageStackDataType: The loaded image stack data.

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
                        f"The file {path} must contain exactly one array, but found {len(data.files)}."
                    )

                # Get the array
                array_name = data.files[0]
                array = data[array_name]

                # Validate that it a 3D array
                if array.ndim != 3:
                    raise ValueError(f"The array in {path} is not a 3D array.")

                # Wrap the array in an ImageStackDataType object
                return ImageStackDataType(array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading image stack data from {path}: {e}") from e


class NpzImageStackDataSaver(ImageStackDataSink):
    """
    Concrete implementation of ImageStackDataTypeSink for saving `ImageStackDataType` objects to .npz files.

    This class provides functionality to save an `ImageStackDataType` object into a `.npz` file.
    """

    def _send_data(self, data: ImageStackDataType, path: str):
        """
        Saves the `ImageStackDataType` as a `.npz` file at the specified path.

        Parameters:
            data (ImageStackDataType): The image stack data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not an `ImageStackDataType`.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of ImageStackDataType.")

        try:
            # Save the data to an .npz file
            np.savez(path, image_stack=data.data)
        except Exception as e:
            raise IOError(f"Error saving ImageStackDataType to {path}: {e}") from e


class PngImageLoader(ImageDataSource):
    """
    Concrete implementation of ImageDataTypeSource for loading image data from PNG files.

    This class provides functionality to load a PNG image as an `ImageDataType`.
    """

    def _get_data(self, path: str) -> ImageDataType:
        """
        Loads a PNG image from the specified file path and returns it as an `ImageDataType`.

        Parameters:
            path (str): The path to the PNG file.

        Returns:
            ImageDataType: The loaded image data.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the file cannot be opened or does not contain valid image data.
        """
        try:
            # Open the PNG image
            with Image.open(path) as img:
                # Convert the image to grayscale and load it as a NumPy array
                image_array = np.asarray(img.convert("L"))
                # Wrap the array in an ImageDataType object
                return ImageDataType(image_array)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading PNG image from {path}: {e}") from e


class PngImageSaver(ImageDataSink):
    """
    Concrete implementation of ImageDataTypeSink for saving image data to PNG files.

    This class provides functionality to save an `ImageDataType` object as a PNG image.
    """

    def _send_data(self, data: ImageDataType, path: str):
        """
        Saves the `ImageDataType` as a PNG file at the specified path.

        Parameters:
            data (ImageDataType): The image data to be saved.
            path (str): The file path to save the PNG image.

        Raises:
            ValueError: If the provided data is not an `ImageDataType`.
            IOError: If the file cannot be saved.
        """
        if not isinstance(
            data, self.input_data_type()
        ):  # Check if the data type is correct
            raise ValueError("Provided data is not an instance of ImageDataType.")

        try:
            # Convert the NumPy array to a PIL image
            img = Image.fromarray(data.data.astype(np.uint8))
            # Save the image as a PNG
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG image to {path}: {e}") from e


class PNGImageStackSaver(ImageStackDataSink):
    """
    Concrete implementation of ImageDataSink for saving multi-frame image data (ImageStackDataType)
    as sequentially numbered PNG files.

    Each frame in the `ImageStackDataType` is saved as a separate PNG file, with filenames
    numbered sequentially (e.g., "frame_000.png", "frame_001.png", ...).
    """

    def _send_data(self, data: ImageStackDataType, base_path: str):
        """
        Saves the `ImageStackDataType` as sequentially numbered PNG files.

        Parameters:
            data (ImageStackDataType): The image stack data to be saved.
            base_path (str): The base file path to save PNG files. A number will
                             be appended to this path for each frame.

        Raises:
            ValueError: If the provided data is not an `ImageStackDataType`.
            IOError: If any frame cannot be saved.
        """
        if not isinstance(
            data, self.input_data_type()
        ):  # Check if the data type is correct
            raise ValueError("Provided data is not an instance of ImageStackDataType.")

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


class ImageDataRandomGenerator(ImageDataSource):
    """
    A random generator for creating `ImageDataType` objects with random data.
    """

    def _get_data(self, shape: tuple[int, int]) -> ImageDataType:
        """
        Generates a dummy `ImageDataType` with random values.

        Parameters:
            shape (tuple[int, int]): The shape (rows, columns) of the generated image data.

        Returns:
            ImageDataType: A dummy image data object containing a 2D array of random values.

        Raises:
            ValueError: If the provided shape does not have exactly two dimensions.
        """

        # Validate that the shape represents a 2D array
        if len(shape) != 2:
            raise ValueError(
                f"Shape must be a tuple with two dimensions, but got {shape}."
            )
        return ImageDataType(np.random.rand(*shape))


class TwoDGaussianImageGenerator(ImageDataSource):
    """Generates an image with a 2D Gaussian signal with optional rotation."""

    def _get_data(
        self,
        x_0: float | int,  # x position
        y_0: float | int,  # y position
        std_dev: float | tuple[float, float],  # Allow single float or tuple
        amplitude: float,
        angle: float = 0.0,  # Rotation angle in degrees
        image_size: tuple[int, int] = (1024, 1024),  # Default image size
    ) -> ImageDataType:
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
            ImageDataType: A n image with a 2D Gaussian shape.
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

        return ImageDataType(z)


class ParametricImageStackGenerator(ImageStackSource):
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

    def _get_data(self):
        """
        Generates a stack of images with evolving parameters.

        Returns:
            ImageStackDataType: The generated image stack data.
        """
        images = [
            self.image_generator.get_data(
                **{
                    key: self._evaluate_param(key, t)
                    for key in self.parametric_expressions
                },
                **self.image_generator_params,
            )
            for t in self.t_values
        ]
        return ImageStackDataType.from_list(images)

    @property
    def t_values(self):
        t_values = np.linspace(
            self.param_ranges["t"][0], self.param_ranges["t"][1], self.num_frames
        )

        return t_values


class ImageStackRandomGenerator(ImageStackSource):
    """
    A random generator for creating `ImageStackDataType` objects with random data.
    """

    def _get_data(self, shape: tuple[int, int, int]) -> ImageStackDataType:
        """
        Generates a dummy `ImageStackDataType` with random values.

        Parameters:
            shape (tuple[int, int, int]): The shape (slices, rows, columns) of the generated
                                          image stack data.

        Returns:
            ImageStackDataType: A dummy image stack data object containing a 3D array of random values.

        Raises:
            ValueError: If the provided shape does not have exactly three dimensions.
        """
        # Validate that the shape represents a 3D array
        if len(shape) != 3:
            raise ValueError(
                f"Shape must be a tuple with three dimensions, but got {shape}."
            )

        return ImageStackDataType(np.random.rand(*shape))


class ImageStackPayloadRandomGenerator(ImageStackPayloadSource):
    """
    A random generator for producing payloads containing ImageStackDataType and ContextType.
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

    def _get_payload(self, *args, **kwargs) -> tuple[ImageStackDataType, ContextType]:
        """
        Generates and returns a dummy payload.

        The payload consists of:
        - `ImageStackDataType`: A 3D NumPy array with random data (a stack of 2D images).
        - `ContextType`: A dictionary containing dummy contextual information.

        Parameters:
            *args: Additional arguments for customization (not used in this implementation).
            **kwargs: Additional keyword arguments for customization (not used in this implementation).

        Returns:
            tuple[ImageStackDataType, dict]:
                A tuple containing the `ImageStackDataType` with dummy data and a
                `ContextType` dictionary with dummy metadata.
        """
        # Generate a dummy 3D NumPy array (stack of 10 images, each 256x256)
        dummy_stack = np.random.rand(10, 256, 256)  # Example stack of 10 images

        # Wrap the stack in an ImageStackDataType and return the payload
        return ImageStackDataType(dummy_stack), ContextType()
