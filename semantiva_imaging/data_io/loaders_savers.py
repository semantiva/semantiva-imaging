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

from typing import List
from pathlib import Path
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
    Loads a SingleChannelImage from a .npz file.
    """

    @classmethod
    def _get_data(cls, path: str):
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

    @classmethod
    def _send_data(cls, data: SingleChannelImage, path: str) -> None:
        """
        Saves the ``SingleChannelImage`` as a `.npz` file at the specified path.

        Parameters:
            data (SingleChannelImage): The image data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImage``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")

        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".npz"
            # Save the data to an .npz file
            np.savez(path, image=data.data)
        except Exception as e:
            raise IOError(f"Error saving SingleChannelImage to {path}: {e}") from e


class NpzSingleChannelImageStackDataLoader(SingleChannelImageStackSource):
    """
    Loads a SingleChannelImageStack from a .npz file.
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

    @classmethod
    def _send_data(cls, data: SingleChannelImageStack, path: str) -> None:
        """
        Saves the ``SingleChannelImageStack`` as a `.npz` file at the specified path.

        Parameters:
            data (SingleChannelImageStack): The image stack data to be saved.
            path (str): The file path to save the `.npz` file.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImageStack``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )

        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".npz"
            # Save the data to an .npz file
            np.savez(path, image_stack=data.data)
        except Exception as e:
            raise IOError(f"Error saving SingleChannelImageStack to {path}: {e}") from e


class PngSingleChannelImageLoader(SingleChannelImageDataSource):
    """
    Loads a SingleChannelImage from a PNG file.
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
    Saves a SingleChannelImage as a PNG file.
    """

    @classmethod
    def _send_data(cls, data: SingleChannelImage, path: str) -> None:
        """
        Saves the ``SingleChannelImage`` as a PNG file at the specified path.

        Parameters:
            data (SingleChannelImage): The image data to be saved.
            path (str): The file path to save the PNG image.

        Raises:
            ValueError: If the provided data is not a ``SingleChannelImage``.
            IOError: If the file cannot be saved.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")

        try:
            # Ensure extension; use Path.with_suffix to avoid recreating Path
            p = Path(path)
            if p.suffix == "":
                p = p.with_suffix(".png")
                path = str(p)

            # Create parent directories if they don't exist
            p.parent.mkdir(parents=True, exist_ok=True)

            # Convert the NumPy array to a PIL image
            img = Image.fromarray(data.data.astype(np.uint8))
            # Save the image as a PNG
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG image to {path}: {e}") from e


class JpgSingleChannelImageLoader(SingleChannelImageDataSource):
    """Load a SingleChannelImage from a JPEG file."""

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
    """Saves a SingleChannelImage to a JPEG file."""

    @classmethod
    def _send_data(cls, data: SingleChannelImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")
        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".jpg"
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="JPEG")
        except Exception as e:
            raise IOError(f"Error saving JPEG image to {path}: {e}") from e


class TiffSingleChannelImageLoader(SingleChannelImageDataSource):
    """Loads a SingleChannelImage from a TIFF file."""

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
    """Saves a SingleChannelImage to a TIFF file."""

    @classmethod
    def _send_data(cls, data: SingleChannelImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of SingleChannelImage.")
        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".tiff"
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="TIFF")
        except Exception as e:
            raise IOError(f"Error saving TIFF image to {path}: {e}") from e


class JpgRGBImageLoader(RGBImageDataSource):
    """Loads an RGBImage from a JPEG file."""

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
    """Saves an RGBImage to a JPEG file."""

    @classmethod
    def _send_data(cls, data: RGBImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".jpg"
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="JPEG")
        except Exception as e:
            raise IOError(f"Error saving JPEG RGB image to {path}: {e}") from e


class PngRGBImageLoader(RGBImageDataSource):
    """Loads an RGBImage from a PNG file."""

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
    """Saves an RGBImage to a PNG file."""

    @classmethod
    def _send_data(cls, data: RGBImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            # Ensure extension; update Path object in-place when needed
            p = Path(path)
            if p.suffix == "":
                p = p.with_suffix(".png")
                path = str(p)

            # Create parent directories if they don't exist
            p.parent.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG RGB image to {path}: {e}") from e


class TiffRGBImageLoader(RGBImageDataSource):
    """Loads an RGBImage from a TIFF file."""

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
    """Saves an RGBImage to a TIFF file."""

    @classmethod
    def _send_data(cls, data: RGBImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImage.")
        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".tiff"
            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="TIFF")
        except Exception as e:
            raise IOError(f"Error saving TIFF RGB image to {path}: {e}") from e


class PngRGBAImageLoader(RGBAImageDataSource):
    """Loads an RGBAImage from a PNG file."""

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
    """Saves an RGBAImage to a PNG file."""

    @classmethod
    def _send_data(cls, data: RGBAImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImage.")
        try:
            # Ensure extension; update Path object in-place when needed
            p = Path(path)
            if p.suffix == "":
                p = p.with_suffix(".png")
                path = str(p)

            # Create parent directories if they don't exist
            p.parent.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(data.data.astype(np.uint8))
            img.save(path, format="PNG")
        except Exception as e:
            raise IOError(f"Error saving PNG RGBA image to {path}: {e}") from e


# ============================================================================
# Helper Functions for Image Stack Savers
# ============================================================================


def _get_numbered_frame_path(base_path: str, frame_index: int) -> str:
    """
    Generate a numbered file path for a frame in an image stack.

    Intelligently handles file extensions: if the base_path has a recognized
    image extension, it keeps that extension and inserts the frame number
    before it. Otherwise, it appends the frame number and .png extension.

    Examples:
        >>> _get_numbered_frame_path("output/frame", 0)
        'output/frame_000.png'
        >>> _get_numbered_frame_path("output/frame.png", 5)
        'output/frame_005.png'
        >>> _get_numbered_frame_path("output/frame.with.dots", 3)
        'output/frame.with.dots_003.png'

    Parameters:
        base_path (str): The base file path (with or without extension).
        frame_index (int): The frame number to insert.

    Returns:
        str: The numbered file path with proper extension handling.
    """
    base = Path(base_path)
    known_image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}
    suffix = base.suffix.lower()

    if suffix == "" or suffix not in known_image_exts:
        # No recognized extension: append frame number and .png
        return f"{base_path}_{frame_index:03d}.png"
    else:
        # Has recognized extension: insert frame number before extension
        # (e.g., "frame.png" -> "frame_000.png")
        return f"{str(base.with_suffix(''))}_{frame_index:03d}{base.suffix}"


def _save_image_stack_frames(stack_data: np.ndarray, base_path: str) -> None:
    """
    Save all frames from an image stack as individually numbered PNG files.

    This helper function is shared by SingleChannel, RGB, and RGBA image
    stack savers to avoid code duplication. It handles:
    - Directory creation (including parent directories)
    - Frame enumeration and numbering
    - File path generation with intelligent extension handling
    - PIL Image conversion and PNG saving

    Parameters:
        stack_data (np.ndarray): The image stack array to save.
        base_path (str): The base file path for the output files.

    Raises:
        IOError: If any frame cannot be saved.
    """
    # Create parent directories if they don't exist
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    # Save each frame with a numbered filename
    for frame_index, frame in enumerate(stack_data):
        file_path = _get_numbered_frame_path(base_path, frame_index)
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(file_path, format="PNG")


class PNGSingleChannelImageStackSaver(SingleChannelImageStackSink):
    """
    Saves multi-frame SingleChannelImageStack data as sequentially numbered PNG files.
    """

    @classmethod
    def _send_data(cls, data: SingleChannelImageStack, base_path: str) -> None:
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
        if not isinstance(data, cls.input_data_type()):
            raise ValueError(
                "Provided data is not an instance of SingleChannelImageStack."
            )

        try:
            _save_image_stack_frames(data.data, base_path)
        except Exception as e:
            raise IOError(f"Error saving PNG image stack: {e}") from e


class PNGRGBImageStackSaver(RGBImageStackSink):
    """
    Saves multi-frame RGB image data (``RGBImageStack``) as sequentially
    numbered PNG files.
    """

    @classmethod
    def _send_data(cls, data: RGBImageStack, base_path: str) -> None:
        """
        Saves the ``RGBImageStack`` as sequentially numbered PNG files.

        Parameters:
            data (RGBImageStack): The RGB image stack data to be saved.
            base_path (str): The base file path to save PNG files. A number will
                             be appended to this path for each frame.

        Raises:
            ValueError: If the provided data is not a ``RGBImageStack``.
            IOError: If any frame cannot be saved.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImageStack.")

        try:
            _save_image_stack_frames(data.data, base_path)
        except Exception as e:
            raise IOError(f"Error saving PNG RGB image stack: {e}") from e


class PNGRGBAImageStackSaver(RGBAImageStackSink):
    """
    Saves multi-frame RGBA image data (``RGBAImageStack``) as sequentially
    numbered PNG files.
    """

    @classmethod
    def _send_data(cls, data: RGBAImageStack, base_path: str) -> None:
        """
        Saves the ``RGBAImageStack`` as sequentially numbered PNG files.

        Parameters:
            data (RGBAImageStack): The RGBA image stack data to be saved.
            base_path (str): The base file path to save PNG files. A number will
                             be appended to this path for each frame.

        Raises:
            ValueError: If the provided data is not a ``RGBAImageStack``.
            IOError: If any frame cannot be saved.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImageStack.")

        try:
            _save_image_stack_frames(data.data, base_path)
        except Exception as e:
            raise IOError(f"Error saving PNG RGBA image stack: {e}") from e


class SingleChannelImageStackVideoLoader(SingleChannelImageStackSource):
    """Loads a SingleChannelImageStack from an AVI video."""

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
    """Saves a SingleChannelImageStack to an AVI video."""

    @classmethod
    def _send_data(cls, data: SingleChannelImageStack, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
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

        # Ensure extension
        p = Path(path)
        if p.suffix == "":
            path = path + ".avi"

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
                            raise IOError("Failed to write frame to video")
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
                    except Exception as e_:
                        raise e_
                writer = None
                continue

        # If we get here, all codecs failed
        error_msg = f"Could not save video to {path}. Tried codecs: {fourcc_options}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise IOError(error_msg)


class RGBImageStackVideoLoader(RGBImageStackSource):
    """Loads an RGBImageStack from an AVI video."""

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
    """Saves an RGBImageStack to an AVI video."""

    @classmethod
    def _send_data(cls, data: RGBImageStack, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImageStack.")

        # Check if we have data to write
        if len(data.data) == 0:
            raise ValueError("Cannot save empty image stack")

        # List of fourcc codecs to try, in order of preference
        fourcc_options = ["MJPG", "XVID", "mp4v", "X264"]

        h, w = data.data.shape[1:3]
        writer = None
        last_error = None

        # Ensure extension
        p = Path(path)
        if p.suffix == "":
            path = path + ".avi"

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
                            raise IOError("Failed to write frame to video")
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
                    except Exception as e_:
                        raise e_
                writer = None
                continue

        # If we get here, all codecs failed
        error_msg = f"Could not save video to {path}. Tried codecs: {fourcc_options}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise IOError(error_msg)


class AnimatedGifSingleChannelImageStackLoader(SingleChannelImageStackSource):
    """Loads a SingleChannelImageStack from an animated GIF."""

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
    """Saves a SingleChannelImageStack to an animated GIF."""

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImageStack]:  # type: ignore
        return SingleChannelImageStack

    @classmethod
    def _send_data(
        cls,
        data: SingleChannelImageStack,
        path: str,
        loop: int = 0,
        duration: int | None = None,
    ) -> None:
        """
        Save a SingleChannelImageStack to an animated GIF.

        Parameters
        ----------
        data : SingleChannelImageStack
            The image stack to save.
        path : str
            Output file path. If no extension provided, `.gif` is appended.
        loop : int, default 0
            Number of times the animation should loop.
            - 0 (default): infinite looping
            - 1: play once then stop
            - N > 1: repeat N times
        duration : int | None, default None
            Duration of each frame in milliseconds.
            - None (default): all frames have equal display time (PIL default ~100ms)
            - int: all frames display for this many milliseconds
                   (e.g., 200 = 5 FPS, 50 = 20 FPS)

        Raises
        ------
        ValueError
            If data is not a SingleChannelImageStack.
        IOError
            If the file cannot be written.
        """
        if not isinstance(data, cls.input_data_type()):
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

            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".gif"

            if duration is not None:
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    loop=loop,
                    duration=duration,
                )
            else:
                frames[0].save(path, save_all=True, append_images=frames[1:], loop=loop)
        except Exception as e:
            raise IOError(f"Error saving GIF to {path}: {e}") from e


class AnimatedGifRGBImageStackLoader(RGBImageStackSource):
    """Loads an RGBImageStack from an animated GIF."""

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
    """Saves an RGBImageStack to an animated GIF."""

    @classmethod
    def _send_data(
        cls,
        data: RGBImageStack,
        path: str,
        loop: int = 0,
        duration: int | None = None,
    ) -> None:
        """
        Save an RGBImageStack to an animated GIF.

        Parameters
        ----------
        data : RGBImageStack
            The image stack to save.
        path : str
            Output file path. If no extension provided, `.gif` is appended.
        loop : int, default 0
            Number of times the animation should loop.
            - 0 (default): infinite looping
            - 1: play once then stop
            - N > 1: repeat N times
        duration : int | None, default None
            Duration of each frame in milliseconds.
            - None (default): all frames have equal display time (PIL default ~100ms)
            - int: all frames display for this many milliseconds
                   (e.g., 200 = 5 FPS, 50 = 20 FPS)

        Raises
        ------
        ValueError
            If data is not an RGBImageStack.
        IOError
            If the file cannot be written.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBImageStack.")
        try:
            frames = [Image.fromarray(f.astype(np.uint8)) for f in data.data]

            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".gif"

            if duration is not None:
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    loop=loop,
                    duration=duration,
                )
            else:
                frames[0].save(path, save_all=True, append_images=frames[1:], loop=loop)
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

    @classmethod
    def _get_payload(cls) -> Payload:
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

    @classmethod
    def _send_data(cls, data: RGBAImage, path: str) -> None:
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImage.")
        try:
            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".tiff"
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

    @classmethod
    def _send_data(
        cls,
        data: RGBAImageStack,
        path: str,
        loop: int = 0,
        duration: int | None = None,
    ) -> None:
        """
        Save an RGBAImageStack to an animated GIF.

        Parameters
        ----------
        data : RGBAImageStack
            The image stack to save.
        path : str
            Output file path. If no extension provided, `.gif` is appended.
        loop : int, default 0
            Number of times the animation should loop.
            - 0 (default): infinite looping
            - 1: play once then stop
            - N > 1: repeat N times
        duration : int | None, default None
            Duration of each frame in milliseconds.
            - None (default): all frames have equal display time (PIL default ~100ms)
            - int: all frames display for this many milliseconds
                   (e.g., 200 = 5 FPS, 50 = 20 FPS)

        Raises
        ------
        ValueError
            If data is not an RGBAImageStack.
        IOError
            If the file cannot be written.
        """
        if not isinstance(data, cls.input_data_type()):
            raise ValueError("Provided data is not an instance of RGBAImageStack.")
        try:
            frames = [Image.fromarray(f.astype(np.uint8)) for f in data.data]

            # Ensure extension
            p = Path(path)
            if p.suffix == "":
                path = path + ".gif"

            if duration is not None:
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    loop=loop,
                    duration=duration,
                )
            else:
                frames[0].save(path, save_all=True, append_images=frames[1:], loop=loop)
        except Exception as e:
            raise IOError(f"Error saving GIF to {path}: {e}") from e
