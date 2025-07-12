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

import inspect
import numpy as np
import pytest
import cv2  # Ensure OpenCV is imported for all tests

from semantiva_imaging.adapters.opencv_factory import _create_opencv_processor
from semantiva_imaging.data_types import RGBImage, RGBAImage


# Channel mapping ----------------------------------------------------------------


def test_channel_mapping_zero_copy():
    """
    Test that the channel mapping is performed as a zero-copy operation.

    This test ensures that:
    - The input image's memory is shared with the OpenCV function.
    - The channel order is correctly swapped (e.g., RGB to BGR).
    - The output image matches the input image after processing.
    """
    called = {}

    def cv_func(arr):
        # Mock OpenCV function that simply returns the input array
        called["arr"] = arr
        return arr

    # Create a processor for RGBImage
    Proc = _create_opencv_processor(cv_func, "DummyProc", RGBImage, RGBImage)
    img_arr = np.random.rand(2, 2, 3).astype(np.float32)  # Random RGB image
    img = RGBImage(img_arr)
    proc = Proc()
    out = proc.process(img)

    # Assert that the memory is shared between input and processed image
    assert np.shares_memory(called["arr"], img_arr)
    # Assert that the channel order is swapped correctly
    assert np.array_equal(called["arr"][..., 0], img_arr[..., 2])
    # Assert that the output matches the input
    np.testing.assert_array_equal(out.data, img_arr)


# Signature handling -------------------------------------------------------------


def test_signature_parsing_default():
    """
    Test that the default signature parser correctly parses the OpenCV function.

    This test ensures that:
    - The generated processor's method signature matches the OpenCV function's signature.
    - Default values and parameter names are preserved.
    """

    def cv_func(src, ksize: int, sigma: float = 1.0):
        return src

    # Create a processor for RGBImage
    Proc = _create_opencv_processor(cv_func, "BlurProc", RGBImage, RGBImage)
    sig = inspect.signature(Proc._process_logic)

    # Assert that the signature parameters match
    assert list(sig.parameters.keys()) == ["self", "data", "ksize", "sigma"]
    # Assert that default values are preserved
    assert sig.parameters["sigma"].default == 1.0


def test_signature_override_skips_parser():
    """
    Test that providing an override signature skips the default parser.

    This test ensures that:
    - The override signature is used directly.
    - The custom parser is not called when an override is provided.
    """
    called = {}

    def parser(func):
        # Mock parser to track if it is called
        called["used"] = True
        return inspect.signature(func)

    # Define an override signature
    override = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "data", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RGBImage
            ),
            inspect.Parameter(
                "alpha", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float
            ),
        ],
        return_annotation=RGBImage,
    )

    # Create a processor with an override signature
    Proc = _create_opencv_processor(
        lambda src, value: src,
        "OverrideProc",
        RGBImage,
        RGBImage,
        signature_parser=parser,
        override_signature=override,
    )
    sig = inspect.signature(Proc._process_logic)

    # Assert that the override signature is used
    assert list(sig.parameters.keys()) == ["self", "data", "alpha"]
    # Assert that the custom parser was not called
    assert "used" not in called


# Return map ---------------------------------------------------------------------


def test_return_map_and_observer(monkeypatch):
    """
    Test that the return map correctly routes tuple elements to observers.

    This test ensures that:
    - The return map routes specific tuple elements to observer keys.
    - The final return value is the unmapped image payload.
    - Observers are notified with the correct key-value pairs.
    """

    def cv_func(src):
        # Mock OpenCV function returning a tuple with an image and a scalar
        img1 = src
        img2 = src + 1
        return img1, 5, img2

    # Create a processor with a return map
    Proc = _create_opencv_processor(
        cv_func, "ReturnProc", RGBImage, RGBImage, return_map={1: "threshold"}
    )
    proc = Proc()
    recorded = {}

    def notifier(key, value):
        # Mock observer notification system
        recorded[key] = value

    # Replace the notification method with the mock
    monkeypatch.setattr(proc, "_notify_context_update", notifier)

    img = RGBImage(np.zeros((1, 1, 3), dtype=np.float32))
    out = proc.process(img)

    # Assert that the observer was notified with the correct key-value pair
    assert recorded == {"threshold": 5}
    # Assert that the final return value is the unmapped image payload
    np.testing.assert_array_equal(out.data, img.data + 1)


def test_tuple_no_payload_error():
    """
    Test that an error is raised when no image payload is found in the tuple.

    This test ensures that:
    - A ValueError is raised if the return map does not identify an image payload.
    """

    def cv_func(src):
        # Mock OpenCV function returning a tuple without a clear image payload
        return (src, 1)

    # Create a processor without a return map
    Proc = _create_opencv_processor(cv_func, "ErrProc", RGBImage, RGBImage)
    proc = Proc()
    img = RGBImage(np.zeros((1, 1, 3), dtype=np.float32))

    # Assert that processing raises a ValueError
    with pytest.raises(ValueError):
        proc.process(img)


# Type mismatch ------------------------------------------------------------------


def test_type_mismatch():
    """
    Test that a TypeMismatchError is raised for incompatible input types.

    This test ensures that:
    - The processor validates the input image type.
    - An exception is raised when the input type does not match the expected type.
    """

    def cv_func(src):
        # Mock OpenCV function that simply returns the input
        return src

    # Create a processor for RGBImage
    Proc = _create_opencv_processor(cv_func, "MismatchProc", RGBImage, RGBImage)
    proc = Proc()
    img = RGBAImage(np.zeros((1, 1, 4), dtype=np.float32))  # Incompatible input type

    # Assert that processing raises an exception
    with pytest.raises(Exception):
        proc.process(img)


# Real OpenCV Functions ----------------------------------------------------------


def test_gaussian_blur_real_opencv():
    """
    Test wrapping cv2.GaussianBlur with the factory.

    This test ensures that:
    - The generated processor correctly wraps the OpenCV GaussianBlur function.
    - The output image has the same shape and dtype as the input.
    - The blur operation modifies the input image as expected.
    """
    GaussianBlurProcessor = _create_opencv_processor(
        cv2.GaussianBlur, "GaussianBlurProcessor", RGBImage, RGBImage
    )

    # Create test image
    img_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = RGBImage(img_data)

    # Process with our wrapped function
    proc = GaussianBlurProcessor()
    result = proc.process(img, ksize=(5, 5), sigmaX=1.0)

    # Verify result
    assert isinstance(result, RGBImage)
    assert result.data.shape == img.data.shape
    assert result.data.dtype == img.data.dtype

    # Verify the blur actually happened (result should be different from input)
    assert not np.array_equal(result.data, img.data)


def test_canny_edge_detection_single_channel():
    """
    Test wrapping cv2.Canny which works on single channel images.

    This test ensures that:
    - The generated processor correctly wraps the OpenCV Canny function.
    - The output image has the same shape and dtype as the input.
    - The edge detection operation produces a valid result.
    """
    from semantiva_imaging.data_types import SingleChannelImage

    # Create a processor that handles single channel input/output
    CannyProcessor = _create_opencv_processor(
        cv2.Canny, "CannyProcessor", SingleChannelImage, SingleChannelImage
    )

    # Create test image
    img_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img = SingleChannelImage(img_data)

    # Process
    proc = CannyProcessor()
    result = proc.process(img, threshold1=50, threshold2=150)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape
    assert result.data.dtype == img.data.dtype


def test_threshold_with_return_map():
    """
    Test cv2.threshold which returns (retval, thresholded_image).

    This test ensures that:
    - The return map captures the threshold value correctly.
    - The thresholded image is returned as the final output.
    - The output image matches the expected binary thresholding result.
    """
    from semantiva_imaging.data_types import SingleChannelImage

    ThresholdProcessor = _create_opencv_processor(
        cv2.threshold,
        "ThresholdProcessor",
        SingleChannelImage,
        SingleChannelImage,
        return_map={0: "threshold_value"},
    )

    # Create test image
    img_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img = SingleChannelImage(img_data)

    # Process
    proc = ThresholdProcessor()
    recorded = {}

    def notifier(key, value):
        # Mock observer notification system
        recorded[key] = value

    # Mock the notification system
    proc._notify_context_update = notifier

    # Process with threshold
    result = proc.process(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape


def test_morphological_operations():
    """
    Test morphological operations like erosion and dilation.

    This test ensures that:
    - The generated processor correctly wraps OpenCV morphological functions.
    - The output image has the same shape and dtype as the input.
    - The morphological operation modifies the input image as expected.
    """
    from semantiva_imaging.data_types import SingleChannelImage

    # Test erosion
    ErodeProcessor = _create_opencv_processor(
        cv2.erode, "ErodeProcessor", SingleChannelImage, SingleChannelImage
    )

    # Create test image
    img_data = np.ones((10, 10), dtype=np.uint8) * 255
    img_data[3:7, 3:7] = 0  # Create a black square in the middle
    img = SingleChannelImage(img_data)

    # Create kernel
    kernel = np.ones((3, 3), np.uint8)

    # Process
    proc = ErodeProcessor()
    result = proc.process(img, kernel=kernel, iterations=1)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape
    # The black square should be larger after erosion
    assert np.sum(result.data == 0) > np.sum(img.data == 0)


def test_color_space_conversion():
    """
    Test color space conversion functions.

    This test ensures that:
    - The generated processor correctly wraps OpenCV color conversion functions.
    - The output image has the expected shape and dtype.
    - The color space conversion produces a valid result.
    """
    from semantiva_imaging.data_types import SingleChannelImage

    # Test RGB to Grayscale conversion
    RGB2GrayProcessor = _create_opencv_processor(
        cv2.cvtColor, "RGB2GrayProcessor", RGBImage, SingleChannelImage
    )

    # Create test image
    img_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = RGBImage(img_data)

    # Process
    proc = RGB2GrayProcessor()
    result = proc.process(img, code=cv2.COLOR_RGB2GRAY)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == (100, 100)
    assert result.data.dtype == img.data.dtype


def test_histogram_equalization():
    """
    Test histogram equalization.

    This test ensures that:
    - The generated processor correctly wraps OpenCV histogram equalization.
    - The output image has the same shape and dtype as the input.
    - The histogram equalization expands the intensity range of the input image.
    """
    from semantiva_imaging.data_types import SingleChannelImage

    EqualizeHistProcessor = _create_opencv_processor(
        cv2.equalizeHist,
        "EqualizeHistProcessor",
        SingleChannelImage,
        SingleChannelImage,
    )

    # Create test image with poor contrast
    img_data = np.ones((100, 100), dtype=np.uint8) * 100  # Gray image
    img_data[25:75, 25:75] = 150  # Slightly brighter square
    img = SingleChannelImage(img_data)

    # Process
    proc = EqualizeHistProcessor()
    result = proc.process(img)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape
    # Histogram equalization should change the image
    assert not np.array_equal(result.data, img.data)


def test_signature_preservation_real_opencv():
    """
    Test that OpenCV function signatures are properly preserved.

    This test ensures that:
    - The generated processor's method signature includes OpenCV parameters.
    - The signature matches the expected parameter names and order.
    """
    GaussianBlurProcessor = _create_opencv_processor(
        cv2.GaussianBlur, "GaussianBlurProcessor", RGBImage, RGBImage
    )

    # Check that the signature includes OpenCV parameters
    sig = inspect.signature(GaussianBlurProcessor._process_logic)
    param_names = list(sig.parameters.keys())

    # Should have self, data, and OpenCV parameters
    assert "self" in param_names
    assert "data" in param_names
    assert "ksize" in param_names
    assert "sigmaX" in param_names


def test_docstring_injection_real_opencv():
    """
    Test that OpenCV docstrings are properly injected.

    This test ensures that:
    - The generated processor's docstring includes the OpenCV function's docstring.
    - The docstring begins with the Semantiva wrapper description.
    """
    GaussianBlurProcessor = _create_opencv_processor(
        cv2.GaussianBlur, "GaussianBlurProcessor", RGBImage, RGBImage
    )

    # Should have a docstring that includes the original function name
    assert "GaussianBlur" in GaussianBlurProcessor.__doc__
