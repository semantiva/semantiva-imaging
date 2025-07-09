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

from semantiva_imaging.adapters import create_opencv_processor
from semantiva_imaging.data_types import RGBImage, RGBAImage


# Channel mapping ----------------------------------------------------------------


def test_channel_mapping_zero_copy():
    called = {}

    def cv_func(arr):
        called["arr"] = arr
        return arr

    Proc = create_opencv_processor(cv_func, "DummyProc", RGBImage, RGBImage)
    img_arr = np.random.rand(2, 2, 3).astype(np.float32)
    img = RGBImage(img_arr)
    proc = Proc()
    out = proc.process(img)

    assert np.shares_memory(called["arr"], img_arr)
    assert np.array_equal(called["arr"][..., 0], img_arr[..., 2])
    np.testing.assert_array_equal(out.data, img_arr)


# Signature handling -------------------------------------------------------------


def test_signature_parsing_default():
    def cv_func(src, ksize: int, sigma: float = 1.0):
        return src

    Proc = create_opencv_processor(cv_func, "BlurProc", RGBImage, RGBImage)
    sig = inspect.signature(Proc._process_logic)
    assert list(sig.parameters.keys()) == ["self", "img", "ksize", "sigma"]
    assert sig.parameters["sigma"].default == 1.0


def test_signature_override_skips_parser():
    called = {}

    def parser(func):
        called["used"] = True
        return inspect.signature(func)

    override = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "img", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RGBImage
            ),
            inspect.Parameter(
                "alpha", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float
            ),
        ],
        return_annotation=RGBImage,
    )

    Proc = create_opencv_processor(
        lambda src, value: src,
        "OverrideProc",
        RGBImage,
        RGBImage,
        signature_parser=parser,
        override_signature=override,
    )
    sig = inspect.signature(Proc._process_logic)
    assert list(sig.parameters.keys()) == ["self", "img", "alpha"]
    assert "used" not in called


# Return map ---------------------------------------------------------------------


def test_return_map_and_observer(monkeypatch):
    def cv_func(src):
        img1 = src
        img2 = src + 1
        return img1, 5, img2

    Proc = create_opencv_processor(
        cv_func, "ReturnProc", RGBImage, RGBImage, return_map={1: "threshold"}
    )
    proc = Proc()
    recorded = {}

    def notifier(key, value):
        recorded[key] = value

    monkeypatch.setattr(proc, "_notify_context_update", notifier)

    img = RGBImage(np.zeros((1, 1, 3), dtype=np.float32))
    out = proc.process(img)

    assert recorded == {"threshold": 5}
    np.testing.assert_array_equal(out.data, img.data + 1)


def test_tuple_no_payload_error():
    def cv_func(src):
        return (src, 1)

    Proc = create_opencv_processor(cv_func, "ErrProc", RGBImage, RGBImage)
    proc = Proc()
    img = RGBImage(np.zeros((1, 1, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        proc.process(img)


# Type mismatch ------------------------------------------------------------------


def test_type_mismatch():
    def cv_func(src):
        return src

    Proc = create_opencv_processor(cv_func, "MismatchProc", RGBImage, RGBImage)
    proc = Proc()
    img = RGBAImage(np.zeros((1, 1, 4), dtype=np.float32))
    with pytest.raises(Exception):
        proc.process(img)


# Real OpenCV Functions ----------------------------------------------------------


pytest.importorskip("cv2")  # Skip all OpenCV tests if cv2 not available
import cv2


def test_gaussian_blur_real_opencv():
    """Test wrapping cv2.GaussianBlur with the factory."""
    GaussianBlurProcessor = create_opencv_processor(
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
    """Test wrapping cv2.Canny which works on single channel images."""
    from semantiva_imaging.data_types import SingleChannelImage

    # Create a processor that handles single channel input/output
    CannyProcessor = create_opencv_processor(
        cv2.Canny, "CannyProcessor", SingleChannelImage, SingleChannelImage
    )

    # Create test single channel image
    img_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img = SingleChannelImage(img_data)

    # Process with Canny edge detection
    proc = CannyProcessor()
    result = proc.process(img, threshold1=50, threshold2=150)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape
    assert result.data.dtype == img.data.dtype


def test_threshold_with_return_map():
    """Test cv2.threshold which returns (retval, thresholded_image)."""
    from semantiva_imaging.data_types import SingleChannelImage

    ThresholdProcessor = create_opencv_processor(
        cv2.threshold,
        "ThresholdProcessor",
        SingleChannelImage,
        SingleChannelImage,
        return_map={0: "threshold_value"},  # First return value is the threshold
    )

    # Create test single channel image with known values
    img_data = np.array([[100, 200], [50, 150]], dtype=np.uint8)
    img = SingleChannelImage(img_data)

    proc = ThresholdProcessor()
    recorded = {}

    def notifier(key, value):
        recorded[key] = value

    # Mock the notification system
    proc._notify_context_update = notifier

    # Process with threshold
    result = proc.process(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    # Verify the threshold value was captured
    assert "threshold_value" in recorded
    assert recorded["threshold_value"] == 100

    # Verify the thresholded image
    assert isinstance(result, SingleChannelImage)
    expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(result.data, expected)


def test_morphological_operations():
    """Test morphological operations like erosion and dilation."""
    from semantiva_imaging.data_types import SingleChannelImage

    # Test erosion
    ErodeProcessor = create_opencv_processor(
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
    """Test color space conversion functions."""
    from semantiva_imaging.data_types import SingleChannelImage

    # Test RGB to Grayscale conversion
    RGB2GrayProcessor = create_opencv_processor(
        cv2.cvtColor, "RGB2GrayProcessor", RGBImage, SingleChannelImage
    )

    # Create test RGB image
    img_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    img = RGBImage(img_data)

    # Process
    proc = RGB2GrayProcessor()
    result = proc.process(img, code=cv2.COLOR_RGB2GRAY)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == (50, 50)
    assert result.data.dtype == np.uint8


def test_histogram_equalization():
    """Test histogram equalization."""
    from semantiva_imaging.data_types import SingleChannelImage

    EqualizeHistProcessor = create_opencv_processor(
        cv2.equalizeHist,
        "EqualizeHistProcessor",
        SingleChannelImage,
        SingleChannelImage,
    )

    # Create test image with poor contrast
    img_data = np.random.randint(50, 100, (100, 100), dtype=np.uint8)
    img = SingleChannelImage(img_data)

    # Process
    proc = EqualizeHistProcessor()
    result = proc.process(img)

    # Verify result
    assert isinstance(result, SingleChannelImage)
    assert result.data.shape == img.data.shape
    assert result.data.dtype == img.data.dtype

    # Verify histogram equalization expanded the range
    assert result.data.min() < img.data.min()
    assert result.data.max() > img.data.max()


def test_signature_preservation_real_opencv():
    """Test that OpenCV function signatures are properly preserved."""
    GaussianBlurProcessor = create_opencv_processor(
        cv2.GaussianBlur, "GaussianBlurProcessor", RGBImage, RGBImage
    )

    # Check that the signature includes OpenCV parameters
    sig = inspect.signature(GaussianBlurProcessor._process_logic)
    param_names = list(sig.parameters.keys())

    # Should have self, img, and OpenCV parameters
    assert "self" in param_names
    assert "img" in param_names
    assert "ksize" in param_names
    assert "sigmaX" in param_names


def test_docstring_injection_real_opencv():
    """Test that OpenCV docstrings are properly injected."""
    GaussianBlurProcessor = create_opencv_processor(
        cv2.GaussianBlur, "GaussianBlurProcessor", RGBImage, RGBImage
    )

    # Check docstring
    doc = GaussianBlurProcessor.__doc__
    assert doc is not None
    assert 'Semantiva wrapper for "GaussianBlur"' in doc
    # OpenCV functions should have some documentation
    assert len(doc) > 100  # Should have substantial content from OpenCV docs
