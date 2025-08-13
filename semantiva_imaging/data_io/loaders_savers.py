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
from PIL import Image, ImageSequence
import numpy as np
import cv2
import logging
from semantiva.context_processors.context_types import ContextType
from semantiva.pipeline.payload import Payload
from .io import (
    SingleChannelImageDataSource,
    SingleChannelImageStackSource,
    SingleChannelImageDataSink,
    SingleChannelImageStackSink,
    SingleChannelImageStackPayloadSource,
    RGBImageDataSource,
    RGBImageStackSource,
    RGBImageDataSink,
    RGBImageStackSink,
    RGBAImageDataSource,
    RGBAImageStackSource,
    RGBAImageDataSink,
    RGBAImageStackSink,
)
from ..data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
    RGBImage,
    RGBAImage,
    RGBImageStack,
    RGBAImageStack,
)


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


class NpzSingleChannelImageDataSaver(SingleChannelImageDataSink):
    """Save a ``SingleChannelImage`` to an ``.npz`` file."""

    def _send_data(self, data: SingleChannelImage, path: str) -> None:
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
    def _get_data(cls, path: str) -> SingleChannelImageStack:
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


class NpzSingleChannelImageStackDataSaver(SingleChannelImageStackSink):
    """Save a ``SingleChannelImageStack`` to an ``.npz`` file."""

    def _send_data(self, data: SingleChannelImageStack, path: str) -> None:
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


class PngSingleChannelImageLoader(SingleChannelImageDataSource):
    """
    Concrete implementation of ImageDataTypeSource for loading image data from PNG files.

    This class provides functionality to load a PNG image as a :class:`SingleChannelImage`.
    """

    @classmethod
    def _get_data(cls, path: str) -> SingleChannelImage:
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


class PngSingleChannelImageSaver(SingleChannelImageDataSink):
    """
    Concrete implementation of ImageDataTypeSink for saving image data to PNG files.

    This class provides functionality to save a ``SingleChannelImage`` object as a PNG image.
    """

    def _send_data(self, data: SingleChannelImage, path: str) -> None:
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


class JpgSingleChannelImageLoader(SingleChannelImageDataSource):
    """Load a :class:`SingleChannelImage` from a JPEG file."""

    @classmethod
    def _get_data(cls, path: str) -> SingleChannelImage:
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"))
                return SingleChannelImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading JPEG image from {path}: {e}") from e


class JpgSingleChannelImageSaver(SingleChannelImageDataSink):
    """Save a :class:`SingleChannelImage` to a JPEG file."""

    def _send_data(self, data: SingleChannelImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="JPEG")
        except Exception as e:
            raise IOError(f"Error saving JPEG image to {path}: {e}") from e


class TiffSingleChannelImageLoader(SingleChannelImageDataSource):
    """Load a :class:`SingleChannelImage` from a TIFF file."""

    @classmethod
    def _get_data(cls, path: str) -> SingleChannelImage:
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"))
                return SingleChannelImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading TIFF image from {path}: {e}") from e


class TiffSingleChannelImageSaver(SingleChannelImageDataSink):
    """Save a :class:`SingleChannelImage` to a TIFF file."""

    def _send_data(self, data: SingleChannelImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="TIFF")
        except Exception as e:
            raise IOError(f"Error saving TIFF image to {path}: {e}") from e


class JpgRGBImageLoader(RGBImageDataSource):
    """Load an :class:`RGBImage` from a JPEG file."""

    @classmethod
    def _get_data(cls, path: str) -> RGBImage:
        try:
            with Image.open(path) as img:
                has_alpha = "A" in img.getbands()
                if has_alpha:
                    logging.warning("Alpha channel dropped")
                arr = np.asarray(img.convert("RGB"))
                return RGBImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading JPEG RGB image from {path}: {e}") from e


class JpgRGBImageSaver(RGBImageDataSink):
    """Save an :class:`RGBImage` to a JPEG file."""

    def _send_data(self, data: RGBImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="JPEG")
        except Exception as e:
            raise IOError(f"Error saving JPEG RGB image to {path}: {e}") from e


class PngRGBImageLoader(RGBImageDataSource):
    """Load an :class:`RGBImage` from a PNG file."""

    @classmethod
    def _get_data(cls, path: str) -> RGBImage:
        try:
            with Image.open(path) as img:
                has_alpha = "A" in img.getbands()
                if has_alpha:
                    logging.warning("Alpha channel dropped")
                arr = np.asarray(img.convert("RGB"))
                return RGBImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading PNG RGB image from {path}: {e}") from e


class PngRGBImageSaver(RGBImageDataSink):
    """Save an :class:`RGBImage` to a PNG file."""

    def _send_data(self, data: RGBImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG RGB image to {path}: {e}") from e


class TiffRGBImageLoader(RGBImageDataSource):
    """Load an :class:`RGBImage` from a TIFF file."""

    @classmethod
    def _get_data(cls, path: str) -> RGBImage:
        try:
            with Image.open(path) as img:
                has_alpha = "A" in img.getbands()
                if has_alpha:
                    logging.warning("Alpha channel dropped")
                arr = np.asarray(img.convert("RGB"))
                return RGBImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading TIFF RGB image from {path}: {e}") from e


class TiffRGBImageSaver(RGBImageDataSink):
    """Save an :class:`RGBImage` to a TIFF file."""

    def _send_data(self, data: RGBImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="TIFF")
        except Exception as e:
            raise IOError(f"Error saving TIFF RGB image to {path}: {e}") from e


class PngRGBAImageLoader(RGBAImageDataSource):
    """Load an :class:`RGBAImage` from a PNG file."""

    @classmethod
    def _get_data(cls, path: str) -> RGBAImage:
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGBA"))
                return RGBAImage(arr)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading PNG RGBA image from {path}: {e}") from e


class PngRGBAImageSaver(RGBAImageDataSink):
    """Save an :class:`RGBAImage` to a PNG file."""

    def _send_data(self, data: RGBAImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG RGBA image to {path}: {e}") from e


class PNGSingleChannelImageStackSaver(SingleChannelImageStackSink):
    """
    Concrete implementation of ImageDataSink for saving multi-frame image data (``SingleChannelImageStack``)
    as sequentially numbered PNG files.

    Each frame in the ``SingleChannelImageStack`` is saved as a separate PNG
    file with filenames numbered sequentially (e.g., ``frame_000.png``).
    """

    def _send_data(self, data: SingleChannelImageStack, base_path: str) -> None:
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


class SingleChannelImageStackVideoLoader(SingleChannelImageStackSource):
    """Load a :class:`SingleChannelImageStack` from an AVI video."""

    @classmethod
    def _get_data(cls, path: str) -> SingleChannelImageStack:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"File not found: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        if not frames:
            raise ValueError(f"No frames found in {path}")
        return SingleChannelImageStack(np.stack(frames))


class SingleChannelImageStackAVISaver(SingleChannelImageStackSink):
    """Save a :class:`SingleChannelImageStack` to an AVI video."""

    def _send_data(self, data: SingleChannelImageStack, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )

        # Check if we have data to write
        if len(data.data) == 0:
            raise ValueError("Cannot save empty image stack")

        # List of fourcc codecs to try, in order of preference
        fourcc_options = ["MJPG", "XVID", "mp4v", "X264"]

        h, w = data.data.shape[1:]
        writer = None
        last_error = None

        for fourcc_str in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)  # type: ignore
                writer = cv2.VideoWriter(
                    path, fourcc, 1.0, (w, h), False
                )  # False for grayscale

                if not writer.isOpened():
                    if writer:
                        writer.release()
                    continue

                # Test writing one frame to see if this codec actually works
                test_frame = data.data[0].astype(np.uint8)

                # Check frame dimensions and type
                if test_frame.shape != (h, w) or test_frame.dtype != np.uint8:
                    raise ValueError(
                        f"Frame preparation failed: expected {(h, w)} uint8, got {test_frame.shape} {test_frame.dtype}"
                    )

                success = writer.write(test_frame)
                if success:
                    # This codec works, write the remaining frames
                    for frame in data.data[1:]:
                        frame_uint8 = frame.astype(np.uint8)
                        success = writer.write(frame_uint8)
                        if not success:
                            raise IOError(f"Failed to write frame to video")
                    writer.release()
                    return  # Success!

                # If we get here, the write failed
                if writer:
                    writer.release()
                writer = None

            except Exception as e:
                last_error = e
                if writer:
                    try:
                        writer.release()
                    except:
                        pass
                writer = None
                continue

        # If we get here, all codecs failed
        error_msg = f"Could not save video to {path}. Tried codecs: {fourcc_options}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise IOError(error_msg)


class RGBImageStackVideoLoader(RGBImageStackSource):
    """Load a :class:`RGBImageStack` from an AVI video."""

    @classmethod
    def _get_data(cls, path: str) -> RGBImageStack:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"File not found: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()
        if not frames:
            raise ValueError(f"No frames found in {path}")
        return RGBImageStack(np.stack(frames))


class RGBImageStackAVISaver(RGBImageStackSink):
    """Save an :class:`RGBImageStack` to an AVI video."""

    def _send_data(self, data: RGBImageStack, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImageStack.")

        # Check if we have data to write
        if len(data.data) == 0:
            raise ValueError("Cannot save empty image stack")

        # List of fourcc codecs to try, in order of preference
        fourcc_options = ["MJPG", "XVID", "mp4v", "X264"]

        h, w = data.data.shape[1:3]
        writer = None
        last_error = None

        for fourcc_str in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)  # type: ignore
                writer = cv2.VideoWriter(path, fourcc, 1.0, (w, h), True)

                if not writer.isOpened():
                    if writer:
                        writer.release()
                    continue

                # Test writing one frame to see if this codec actually works
                test_frame = data.data[0]
                bgr = cv2.cvtColor(test_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

                # Check frame dimensions and type
                if bgr.shape[:2] != (h, w) or bgr.dtype != np.uint8:
                    raise ValueError(
                        f"Frame preparation failed: expected {(h, w)} uint8, got {bgr.shape[:2]} {bgr.dtype}"
                    )

                success = writer.write(bgr)
                if success:
                    # This codec works, write the remaining frames
                    for frame in data.data[1:]:
                        bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        success = writer.write(bgr)
                        if not success:
                            raise IOError(f"Failed to write frame to video")
                    writer.release()
                    return  # Success!

                # If we get here, the write failed
                if writer:
                    writer.release()
                writer = None

            except Exception as e:
                last_error = e
                if writer:
                    try:
                        writer.release()
                    except:
                        pass
                writer = None
                continue

        # If we get here, all codecs failed
        error_msg = f"Could not save video to {path}. Tried codecs: {fourcc_options}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise IOError(error_msg)


class AnimatedGifSingleChannelImageStackLoader(SingleChannelImageStackSource):
    """Load a :class:`SingleChannelImageStack` from an animated GIF.

    Loads animated GIF files and converts them to grayscale single channel image stacks.
    Each frame of the GIF is converted to grayscale using PIL's "L" mode conversion.
    """

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImageStack]:
        return SingleChannelImageStack

    @classmethod
    def _get_data(cls, path: str) -> SingleChannelImageStack:
        try:
            with Image.open(path) as img:
                frames = [
                    np.asarray(frame.convert("L"))  # Convert each frame to grayscale
                    for frame in ImageSequence.Iterator(img)
                ]
            if not frames:
                raise ValueError(f"No frames found in {path}")
            return SingleChannelImageStack(np.stack(frames))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading GIF from {path}: {e}") from e


class AnimatedGifSingleChannelImageStackSaver(SingleChannelImageStackSink):
    """Save a :class:`SingleChannelImageStack` to an animated GIF.

    Saves single channel image stacks as animated GIF files. Each frame is converted
    from grayscale to RGB by replicating the single channel across R, G, and B channels.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImageStack]:  # type: ignore
        return SingleChannelImageStack

    def _send_data(self, data: SingleChannelImageStack, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )
        try:
            # Convert single channel frames to RGB by replicating the channel
            frames = []
            for frame in data.data:
                # Convert single channel to RGB by stacking 3 times
                rgb_frame = np.stack([frame, frame, frame], axis=-1)
                frames.append(Image.fromarray(rgb_frame.astype(np.uint8)))
            frames[0].save(path, save_all=True, append_images=frames[1:])
        except Exception as e:
            raise IOError(f"Error saving GIF to {path}: {e}") from e


class AnimatedGifRGBImageStackLoader(RGBImageStackSource):
    """Load an :class:`RGBImageStack` from an animated GIF."""

    @classmethod
    def _get_data(cls, path: str) -> RGBImageStack:
        try:
            with Image.open(path) as img:
                frames = [
                    np.asarray(frame.convert("RGB"))
                    for frame in ImageSequence.Iterator(img)
                ]
            if not frames:
                raise ValueError(f"No frames found in {path}")
            return RGBImageStack(np.stack(frames))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading GIF from {path}: {e}") from e


class AnimatedGifRGBImageStackSaver(RGBImageStackSink):
    """Save an :class:`RGBImageStack` to an animated GIF."""

    def _send_data(self, data: RGBImageStack, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImageStack.")
        try:
            frames = [Image.fromarray(f.astype(np.uint8)) for f in data.data]
            frames[0].save(path, save_all=True, append_images=frames[1:])
        except Exception as e:
            raise IOError(f"Error saving GIF to {path}: {e}") from e


class SingleChannelImageRandomGenerator(SingleChannelImageDataSource):
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


class TwoDGaussianSingleChannelImageGenerator(SingleChannelImageDataSource):
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

    def _get_data(self):
        """
        Generates a stack of images with evolving parameters.

        Returns:
            SingleChannelImageStack: The generated image stack data.
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

    def _get_payload(self, *args, **kwargs) -> Payload:
        """
        Generates and returns a dummy payload.

        The payload consists of:
        - ``SingleChannelImageStack``: A 3D NumPy array with random data (a stack of 2D images).
        - `ContextType`: A dictionary containing dummy contextual information.

        Parameters:
            *args: Additional arguments for customization (not used in this implementation).
            **kwargs: Additional keyword arguments for customization (not used in this implementation).

        Returns:
            Payload: A Payload containing the ``SingleChannelImageStack`` with dummy data and a
                ``ContextType`` with dummy metadata.
        """
        # Generate a dummy 3D NumPy array (stack of 10 images, each 256x256)
        dummy_stack = np.random.rand(10, 256, 256)  # Example stack of 10 images

        # Wrap into Payload and return
        return Payload(SingleChannelImageStack(dummy_stack), ContextType())


class TiffRGBAImageLoader(RGBAImageDataSource):
    """Load an :class:`RGBAImage` from a TIFF file."""

    @classmethod
    def _get_data(cls, path: str):
        try:
            with Image.open(path) as img:
                rgba_img = img.convert("RGBA")
                return RGBAImage(np.asarray(rgba_img))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading TIFF image from {path}: {e}") from e


class TiffRGBAImageSaver(RGBAImageDataSink):
    """Save an :class:`RGBAImage` to a TIFF file."""

    def _send_data(self, data: RGBAImage, path: str) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImage.")
        try:
            img = Image.fromarray(data.data.astype(np.uint8), mode="RGBA")
            img.save(path, format="TIFF")
        except Exception as e:
            raise IOError(f"Error saving TIFF image to {path}: {e}") from e


class AnimatedGifRGBAImageStackLoader(RGBAImageStackSource):
    """Load an :class:`RGBAImageStack` from an animated GIF."""

    @classmethod
    def _get_data(cls, path: str):
        try:
            with Image.open(path) as img:
                frames = [
                    np.asarray(frame.convert("RGBA"))
                    for frame in ImageSequence.Iterator(img)
                ]
            if not frames:
                raise ValueError(f"No frames found in {path}")
            return RGBAImageStack(np.stack(frames))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Error loading GIF from {path}: {e}") from e


class AnimatedGifRGBAImageStackSaver(RGBAImageStackSink):
    """Save an :class:`RGBAImageStack` to an animated GIF."""

    def _send_data(self, data: RGBAImageStack, path: str, *args, **kwargs) -> None:
        if not isinstance(data, self.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImageStack.")
        try:
            frames = [Image.fromarray(f.astype(np.uint8)) for f in data.data]
            frames[0].save(path, save_all=True, append_images=frames[1:])
        except Exception as e:
            raise IOError(f"Error saving GIF to {path}: {e}") from e
