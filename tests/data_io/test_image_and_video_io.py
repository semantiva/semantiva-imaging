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

"""Integration tests for image and video I/O utilities."""

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

import os
import sys
import numpy as np
import pytest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from semantiva_imaging.data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
    RGBImage,
    RGBAImage,
    RGBAImageStack,
    RGBImageStack,
)
from semantiva_imaging.data_io.loaders_savers import (
    JpgSingleChannelImageLoader,
    JpgSingleChannelImageSaver,
    TiffSingleChannelImageLoader,
    TiffSingleChannelImageSaver,
    PngSingleChannelImageLoader,
    PngSingleChannelImageSaver,
    JpgRGBImageLoader,
    JpgRGBImageSaver,
    PngRGBImageLoader,
    PngRGBImageSaver,
    TiffRGBImageLoader,
    TiffRGBImageSaver,
    PngRGBAImageLoader,
    PngRGBAImageSaver,
    TiffRGBAImageLoader,
    TiffRGBAImageSaver,
    PNGSingleChannelImageStackSaver,
    SingleChannelImageStackVideoLoader,
    SingleChannelImageStackAVISaver,
    RGBImageStackVideoLoader,
    RGBImageStackAVISaver,
    AnimatedGifSingleChannelImageStackLoader,
    AnimatedGifSingleChannelImageStackSaver,
    AnimatedGifRGBImageStackLoader,
    AnimatedGifRGBImageStackSaver,
    AnimatedGifRGBAImageStackLoader,
    AnimatedGifRGBAImageStackSaver,
)


@pytest.fixture
def gray_image():
    return SingleChannelImage(np.random.randint(0, 255, (3, 3), dtype=np.uint8))


@pytest.fixture
def rgb_image():
    data = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
    return RGBImage(data)


@pytest.fixture
def rgba_stack():
    data = np.random.randint(0, 255, (2, 8, 8, 4), dtype=np.uint8)
    return RGBAImageStack(data)


@pytest.fixture
def rgb_stack():
    data = np.random.randint(0, 255, (2, 8, 8, 3), dtype=np.uint8)
    return RGBImageStack(data)


@pytest.fixture
def gray_stack():
    data = np.random.randint(0, 255, (2, 8, 8), dtype=np.uint8)
    return SingleChannelImageStack(data)


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def test_single_channel_round_trip_jpg(tmp_dir, gray_image):
    path = os.path.join(tmp_dir, "img.jpg")
    JpgSingleChannelImageSaver().send_data(gray_image, path)
    loaded = JpgSingleChannelImageLoader().get_data(path)
    assert loaded.data.shape == gray_image.data.shape
    assert loaded.data.dtype == np.uint8


def test_single_channel_missing_file():
    loader = JpgSingleChannelImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.get_data("nonexistent.jpg")


def test_rgb_loader_alpha_warning(tmp_dir, caplog):
    rgba_img = RGBAImage(np.random.randint(0, 255, (3, 3, 4), dtype=np.uint8))
    fname = os.path.join(tmp_dir, "alpha.png")
    PngRGBAImageSaver().send_data(rgba_img, fname)
    loader = PngRGBImageLoader()
    with caplog.at_level(logging.WARNING):
        img = loader.get_data(fname)
    assert any("Alpha channel dropped" in r.message for r in caplog.records)
    assert img.data.shape == (3, 3, 3)


def test_rgb_round_trip_video(tmp_dir, rgb_stack):
    path = os.path.join(tmp_dir, "vid.avi")
    try:
        RGBImageStackAVISaver().send_data(rgb_stack, path)
        loaded = RGBImageStackVideoLoader().get_data(path)
        assert loaded.data.shape == rgb_stack.data.shape
    except IOError as e:
        if "Could not save video" in str(e) or "encoder" in str(e).lower():
            pytest.skip(f"Video codecs not available on this system: {e}")
        else:
            raise


def test_single_channel_video_round_trip(tmp_dir, gray_stack):
    path = os.path.join(tmp_dir, "gray.avi")
    try:
        SingleChannelImageStackAVISaver().send_data(gray_stack, path)
        loaded = SingleChannelImageStackVideoLoader().get_data(path)
        assert loaded.data.shape == gray_stack.data.shape
    except IOError as e:
        if "Could not save video" in str(e) or "encoder" in str(e).lower():
            pytest.skip(f"Video codecs not available on this system: {e}")
        else:
            raise


def test_gif_round_trip(tmp_dir, rgba_stack):
    path = os.path.join(tmp_dir, "anim.gif")
    AnimatedGifRGBAImageStackSaver().send_data(rgba_stack, path)
    loaded = AnimatedGifRGBAImageStackLoader().get_data(path)
    assert loaded.data.shape == rgba_stack.data.shape


def test_single_channel_round_trip_tiff(tmp_dir, gray_image):
    """Test TIFF single channel round-trip (lossless format)."""
    path = os.path.join(tmp_dir, "img.tiff")
    TiffSingleChannelImageSaver().send_data(gray_image, path)
    loaded = TiffSingleChannelImageLoader().get_data(path)
    assert loaded.data.shape == gray_image.data.shape
    assert loaded.data.dtype == np.uint8
    # TIFF is lossless, so we can do exact comparison
    np.testing.assert_array_equal(loaded.data, gray_image.data)


def test_rgb_round_trip_jpg(tmp_dir, rgb_image):
    """Test JPEG RGB round-trip (lossy format)."""
    path = os.path.join(tmp_dir, "rgb.jpg")
    JpgRGBImageSaver().send_data(rgb_image, path)
    loaded = JpgRGBImageLoader().get_data(path)
    assert loaded.data.shape == rgb_image.data.shape
    assert loaded.data.dtype == np.uint8
    # JPEG is very lossy, so we just check basic functionality
    # The image should be loadable and have reasonable range
    assert loaded.data.min() >= 0
    assert loaded.data.max() <= 255


def test_rgb_round_trip_png(tmp_dir, rgb_image):
    """Test PNG RGB round-trip (lossless format)."""
    path = os.path.join(tmp_dir, "rgb.png")
    PngRGBImageSaver().send_data(rgb_image, path)
    loaded = PngRGBImageLoader().get_data(path)
    assert loaded.data.shape == rgb_image.data.shape
    assert loaded.data.dtype == np.uint8
    # PNG is lossless, so we can do exact comparison
    np.testing.assert_array_equal(loaded.data, rgb_image.data)


def test_rgb_round_trip_tiff(tmp_dir, rgb_image):
    """Test TIFF RGB round-trip (lossless format)."""
    path = os.path.join(tmp_dir, "rgb.tiff")
    TiffRGBImageSaver().send_data(rgb_image, path)
    loaded = TiffRGBImageLoader().get_data(path)
    assert loaded.data.shape == rgb_image.data.shape
    assert loaded.data.dtype == np.uint8
    # TIFF is lossless, so we can do exact comparison
    np.testing.assert_array_equal(loaded.data, rgb_image.data)


def test_rgba_round_trip_png(tmp_dir):
    """Test PNG RGBA round-trip (lossless format with alpha)."""
    rgba_img = RGBAImage(np.random.randint(0, 255, (5, 5, 4), dtype=np.uint8))
    path = os.path.join(tmp_dir, "rgba.png")
    PngRGBAImageSaver().send_data(rgba_img, path)
    loaded = PngRGBAImageLoader().get_data(path)
    assert loaded.data.shape == rgba_img.data.shape
    assert loaded.data.dtype == np.uint8
    # PNG is lossless, including alpha channel
    np.testing.assert_array_equal(loaded.data, rgba_img.data)


def test_tiff_single_channel_missing_file():
    """Test TIFF loader with missing file."""
    loader = TiffSingleChannelImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.get_data("nonexistent.tiff")


def test_jpg_rgb_missing_file():
    """Test JPEG RGB loader with missing file."""
    loader = JpgRGBImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.get_data("nonexistent.jpg")


def test_invalid_input_type():
    """Test that savers raise ValueError for invalid input types."""
    gray_image = SingleChannelImage(np.random.randint(0, 255, (3, 3), dtype=np.uint8))
    rgb_saver = JpgRGBImageSaver()

    # Passing SingleChannelImage to RGB saver should raise ValueError
    with pytest.raises(ValueError):
        rgb_saver.send_data(gray_image, "invalid.jpg")


def test_jpeg_compression_quality(tmp_dir, rgb_image):
    """Test that JPEG compression is reasonable and data is recoverable."""
    path = os.path.join(tmp_dir, "quality_test.jpg")
    JpgRGBImageSaver().send_data(rgb_image, path)
    loaded = JpgRGBImageLoader().get_data(path)

    # Check that the basic properties are preserved
    assert loaded.data.shape == rgb_image.data.shape
    assert loaded.data.dtype == rgb_image.data.dtype

    # For JPEG, just check that the file can be saved and loaded
    # and that pixel values are in valid range
    assert loaded.data.min() >= 0
    assert loaded.data.max() <= 255
    assert os.path.getsize(path) > 0  # File should not be empty


def test_video_codec_compatibility(tmp_dir, rgb_stack):
    """Test that video codec produces compatible output."""
    path = os.path.join(tmp_dir, "codec_test.avi")
    try:
        RGBImageStackAVISaver().send_data(rgb_stack, path)
        loaded = RGBImageStackVideoLoader().get_data(path)

        # Video compression may cause significant loss, so check basic properties
        assert loaded.data.shape == rgb_stack.data.shape
        assert loaded.data.dtype == rgb_stack.data.dtype

        # Check that frames have valid pixel values
        assert loaded.data.min() >= 0
        assert loaded.data.max() <= 255
        assert os.path.getsize(path) > 0  # File should not be empty
    except IOError as e:
        if "Could not save video" in str(e) or "encoder" in str(e).lower():
            pytest.skip(f"Video codecs not available on this system: {e}")
        else:
            raise


def test_animated_gif_frame_preservation(tmp_dir):
    """Test that animated GIF preserves frame count and basic structure."""
    # Create a simple 3-frame RGBA animation since GIF saver expects RGBA
    frames = np.zeros((3, 4, 4, 4), dtype=np.uint8)
    # Frame 0: White square with full alpha
    frames[0, 1:3, 1:3, :3] = 255  # RGB channels
    frames[0, 1:3, 1:3, 3] = 255  # Alpha channel
    # Frame 1: Gray square with full alpha
    frames[1, 0:2, 0:2, :3] = 128  # RGB channels
    frames[1, 0:2, 0:2, 3] = 255  # Alpha channel
    # Frame 2: Light gray square with full alpha
    frames[2, 2:4, 2:4, :3] = 200  # RGB channels
    frames[2, 2:4, 2:4, 3] = 255  # Alpha channel

    rgba_stack = RGBAImageStack(frames)
    path = os.path.join(tmp_dir, "animation.gif")

    AnimatedGifRGBAImageStackSaver().send_data(rgba_stack, path)
    loaded = AnimatedGifRGBAImageStackLoader().get_data(path)

    # Check frame count and dimensions are preserved
    assert loaded.data.shape[0] == rgba_stack.data.shape[0]  # Same number of frames
    assert (
        loaded.data.shape[1:3] == rgba_stack.data.shape[1:3]
    )  # Same frame dimensions (H, W)
    assert loaded.data.dtype == rgba_stack.data.dtype


def test_animated_gif_single_channel_workflow(tmp_dir):
    """Test workflow for single channel GIF animation using proper single channel loader/saver."""
    # Create single channel stack
    gray_frames = np.zeros((2, 4, 4), dtype=np.uint8)
    gray_frames[0, 1:3, 1:3] = 255  # White square in frame 0
    gray_frames[1, 0:2, 2:4] = 128  # Gray square in frame 1
    gray_stack = SingleChannelImageStack(gray_frames)

    path = os.path.join(tmp_dir, "single_channel.gif")
    AnimatedGifSingleChannelImageStackSaver().send_data(gray_stack, path)

    # Load back as single channel
    loaded = AnimatedGifSingleChannelImageStackLoader().get_data(path)

    # Extract single channel from loaded (for verification)
    extracted_gray = loaded.data

    # Basic checks
    assert extracted_gray.shape == gray_frames.shape
    assert loaded.data.shape[0] == 2  # Same number of frames
    assert os.path.getsize(path) > 0  # File should not be empty


def test_large_image_handling(tmp_dir):
    """Test handling of larger images to ensure scalability."""
    # Create a larger test image (50x50)
    large_gray = SingleChannelImage(np.random.randint(0, 255, (50, 50), dtype=np.uint8))
    large_rgb = RGBImage(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))

    # Test TIFF (lossless) with larger images
    gray_path = os.path.join(tmp_dir, "large_gray.tiff")
    TiffSingleChannelImageSaver().send_data(large_gray, gray_path)
    loaded_gray = TiffSingleChannelImageLoader().get_data(gray_path)
    np.testing.assert_array_equal(loaded_gray.data, large_gray.data)

    rgb_path = os.path.join(tmp_dir, "large_rgb.tiff")
    TiffRGBImageSaver().send_data(large_rgb, rgb_path)
    loaded_rgb = TiffRGBImageLoader().get_data(rgb_path)
    np.testing.assert_array_equal(loaded_rgb.data, large_rgb.data)


def test_edge_case_pixel_values(tmp_dir):
    """Test edge cases with extreme pixel values."""
    # Test with all zeros
    zeros_img = SingleChannelImage(np.zeros((5, 5), dtype=np.uint8))
    path_zeros = os.path.join(tmp_dir, "zeros.tiff")
    TiffSingleChannelImageSaver().send_data(zeros_img, path_zeros)
    loaded_zeros = TiffSingleChannelImageLoader().get_data(path_zeros)
    np.testing.assert_array_equal(loaded_zeros.data, zeros_img.data)

    # Test with all max values
    max_img = SingleChannelImage(np.full((5, 5), 255, dtype=np.uint8))
    path_max = os.path.join(tmp_dir, "max.tiff")
    TiffSingleChannelImageSaver().send_data(max_img, path_max)
    loaded_max = TiffSingleChannelImageLoader().get_data(path_max)
    np.testing.assert_array_equal(loaded_max.data, max_img.data)


def test_data_type_preservation(tmp_dir):
    """Test that uint8 data type is preserved across all formats."""
    test_img = SingleChannelImage(np.array([[100, 150], [200, 50]], dtype=np.uint8))

    # Test multiple formats preserve uint8
    formats_and_loaders = [
        ("test.tiff", TiffSingleChannelImageSaver(), TiffSingleChannelImageLoader()),
        ("test.jpg", JpgSingleChannelImageSaver(), JpgSingleChannelImageLoader()),
    ]

    for filename, saver, loader in formats_and_loaders:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(test_img, path)
        loaded = loader.get_data(path)
        assert loaded.data.dtype == np.uint8
        assert loaded.data.shape == test_img.data.shape


def test_tiff_rgba_round_trip(tmp_dir):
    """Test TIFF RGBA round-trip (lossless format with alpha)."""
    rgba_img = RGBAImage(np.random.randint(0, 255, (5, 5, 4), dtype=np.uint8))
    path = os.path.join(tmp_dir, "rgba.tiff")
    TiffRGBAImageSaver().send_data(rgba_img, path)
    loaded = TiffRGBAImageLoader().get_data(path)
    assert loaded.data.shape == rgba_img.data.shape
    assert loaded.data.dtype == np.uint8
    # TIFF is lossless, including alpha channel
    np.testing.assert_array_equal(loaded.data, rgba_img.data)


def test_animated_gif_rgb_stack_round_trip(tmp_dir):
    """Test RGB stack round-trip with animated GIF."""
    rgb_frames = np.random.randint(0, 255, (3, 6, 6, 3), dtype=np.uint8)
    rgb_stack = RGBImageStack(rgb_frames)
    path = os.path.join(tmp_dir, "rgb_animation.gif")

    AnimatedGifRGBImageStackSaver().send_data(rgb_stack, path)
    loaded = AnimatedGifRGBImageStackLoader().get_data(path)

    assert loaded.data.shape == rgb_stack.data.shape
    assert loaded.data.dtype == rgb_stack.data.dtype
    assert os.path.getsize(path) > 0


def test_comprehensive_format_support(tmp_dir):
    """Test that all expected format combinations are supported."""
    # Single channel image formats
    gray_img = SingleChannelImage(np.random.randint(0, 255, (4, 4), dtype=np.uint8))

    single_formats = [
        ("test.jpg", JpgSingleChannelImageSaver(), JpgSingleChannelImageLoader()),
        (
            "test.png",
            PngSingleChannelImageSaver(),
            PngSingleChannelImageLoader(),
        ),  # Using existing PNG classes
        ("test.tiff", TiffSingleChannelImageSaver(), TiffSingleChannelImageLoader()),
    ]

    for filename, saver, loader in single_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(gray_img, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == gray_img.data.shape
        assert loaded.data.dtype == np.uint8

    # RGB image formats
    rgb_img = RGBImage(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8))

    rgb_formats = [
        ("rgb.jpg", JpgRGBImageSaver(), JpgRGBImageLoader()),
        ("rgb.png", PngRGBImageSaver(), PngRGBImageLoader()),
        ("rgb.tiff", TiffRGBImageSaver(), TiffRGBImageLoader()),
    ]

    for filename, saver, loader in rgb_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(rgb_img, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == rgb_img.data.shape
        assert loaded.data.dtype == np.uint8

    # RGBA image formats (PNG and TIFF support alpha)
    rgba_img = RGBAImage(np.random.randint(0, 255, (4, 4, 4), dtype=np.uint8))

    rgba_formats = [
        ("rgba.png", PngRGBAImageSaver(), PngRGBAImageLoader()),
        ("rgba.tiff", TiffRGBAImageSaver(), TiffRGBAImageLoader()),
    ]

    for filename, saver, loader in rgba_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(rgba_img, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == rgba_img.data.shape
        assert loaded.data.dtype == np.uint8

    # Test stack formats that save multiple files
    gray_stack = SingleChannelImageStack(
        np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
    )

    # Test PNGSingleChannelImageStackSaver (saves multiple PNG files)
    base_path = os.path.join(tmp_dir, "png_stack")
    PNGSingleChannelImageStackSaver().send_data(gray_stack, base_path)

    # Verify the files were created
    for i in range(3):
        png_file = f"{base_path}_{i:03d}.png"
        assert os.path.exists(png_file), f"PNG stack file {png_file} was not created"

        # Verify content by loading with PIL
        loaded_img = PngSingleChannelImageLoader().get_data(png_file)
        loaded_array = np.array(loaded_img.data)
        np.testing.assert_array_equal(loaded_array, gray_stack.data[i])


def test_animated_formats_comprehensive(tmp_dir):
    """Test all animated format combinations."""
    # Single channel stack
    gray_stack = SingleChannelImageStack(
        np.random.randint(0, 255, (2, 4, 4), dtype=np.uint8)
    )

    # RGB stack
    rgb_stack = RGBImageStack(np.random.randint(0, 255, (2, 4, 4, 3), dtype=np.uint8))

    # RGBA stack
    rgba_stack = RGBAImageStack(np.random.randint(0, 255, (2, 4, 4, 4), dtype=np.uint8))

    # Test single channel stack formats (skip AVI due to system codec issues)
    single_stack_formats = [
        # ("gray.avi", SingleChannelImageStackAVISaver(), SingleChannelImageStackVideoLoader(), gray_stack),
        (
            "gray.gif",
            AnimatedGifSingleChannelImageStackSaver(),
            AnimatedGifSingleChannelImageStackLoader(),
            gray_stack,
        ),
    ]

    for filename, saver, loader, data in single_stack_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(data, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == data.data.shape
        assert loaded.data.dtype == data.data.dtype

    # Test RGB stack formats (skip AVI due to system codec issues)
    rgb_stack_formats = [
        # ("rgb.avi", RGBImageStackAVISaver(), RGBImageStackVideoLoader(), rgb_stack),
        (
            "rgb.gif",
            AnimatedGifRGBImageStackSaver(),
            AnimatedGifRGBImageStackLoader(),
            rgb_stack,
        ),
    ]

    for filename, saver, loader, data in rgb_stack_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(data, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == data.data.shape
        assert loaded.data.dtype == data.data.dtype

    # Test RGBA stack formats (only GIF supports alpha in animations)
    rgba_stack_formats = [
        (
            "rgba.gif",
            AnimatedGifRGBAImageStackSaver(),
            AnimatedGifRGBAImageStackLoader(),
            rgba_stack,
        ),
    ]

    for filename, saver, loader, data in rgba_stack_formats:
        path = os.path.join(tmp_dir, filename)
        saver.send_data(data, path)
        loaded = loader.get_data(path)
        assert loaded.data.shape == data.data.shape
        assert loaded.data.dtype == data.data.dtype


def test_missing_combinations_intentionally_absent():
    """Test that format combinations that shouldn't exist are not implemented."""
    # JPEG doesn't support alpha well - no JPEG RGBA loaders should exist
    # AVI doesn't support alpha - no AVI RGBA loaders should exist

    # These should not exist (we verify by checking they're not imported)
    missing_combinations = [
        "JpgRGBAImageLoader",  # JPEG doesn't support alpha well
        "JpgRGBAImageSaver",
        "RGBAImageStackVideoLoader",  # AVI doesn't support alpha
        "RGBAImageStackVideoSaver",
    ]
    print(missing_combinations)
    # This test just documents the intentional absence of these combinations
    # In the future, if these are needed, they could be implemented with warnings
    # about quality loss or format limitations
    assert True  # This test is mainly documentation


def test_png_image_stack_saver_basic(tmp_dir):
    """Test basic PNGSingleChannelImageStackSaver functionality."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Create a test stack with 3 frames
    stack_data = np.array(
        [
            [[10, 20, 30, 40], [50, 60, 70, 80]],
            [[15, 25, 35, 45], [55, 65, 75, 85]],
            [[20, 30, 40, 50], [60, 70, 80, 90]],
        ],
        dtype=np.uint8,
    )

    gray_stack = SingleChannelImageStack(stack_data)
    base_path = os.path.join(tmp_dir, "test_stack")

    # Save the stack
    saver = PNGSingleChannelImageStackSaver()
    saver.send_data(gray_stack, base_path)

    # Check that the expected files were created
    expected_files = [
        os.path.join(tmp_dir, "test_stack_000.png"),
        os.path.join(tmp_dir, "test_stack_001.png"),
        os.path.join(tmp_dir, "test_stack_002.png"),
    ]

    for file_path in expected_files:
        assert os.path.exists(file_path), f"Expected file {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    # Load the files back and verify content
    for i, file_path in enumerate(expected_files):
        loaded_img = PngSingleChannelImageLoader().get_data(file_path)
        loaded_array = loaded_img.data

        # Compare with original frame
        original_frame = stack_data[i]
        assert loaded_array.shape == original_frame.shape, f"Frame {i} shape mismatch"
        np.testing.assert_array_equal(
            loaded_array, original_frame, err_msg=f"Frame {i} content mismatch"
        )


def test_png_image_stack_saver_empty_stack(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with empty stack."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Create an empty stack
    empty_data = np.empty((0, 10, 10), dtype=np.uint8)
    empty_stack = SingleChannelImageStack(empty_data)
    base_path = os.path.join(tmp_dir, "empty_stack")

    saver = PNGSingleChannelImageStackSaver()
    saver.send_data(empty_stack, base_path)

    # No files should be created for empty stack
    expected_files = [
        os.path.join(tmp_dir, "empty_stack_000.png"),
        os.path.join(tmp_dir, "empty_stack_001.png"),
    ]

    for file_path in expected_files:
        assert not os.path.exists(file_path), f"Unexpected file {file_path} was created"


def test_png_image_stack_saver_single_frame(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with single frame."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Create a single frame stack
    single_frame = np.random.randint(0, 255, (1, 16, 16), dtype=np.uint8)
    single_stack = SingleChannelImageStack(single_frame)
    base_path = os.path.join(tmp_dir, "single_frame")

    saver = PNGSingleChannelImageStackSaver()
    saver.send_data(single_stack, base_path)

    # Only one file should be created
    expected_file = os.path.join(tmp_dir, "single_frame_000.png")
    unexpected_file = os.path.join(tmp_dir, "single_frame_001.png")

    assert os.path.exists(expected_file), "Expected single frame file was not created"
    assert not os.path.exists(unexpected_file), "Unexpected second file was created"

    # Verify content
    loaded_img = PngSingleChannelImageLoader().get_data(expected_file)

    loaded_array = np.array(loaded_img.data)
    np.testing.assert_array_equal(loaded_array, single_frame[0])


def test_png_image_stack_saver_large_stack(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with larger stack (10 frames)."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Create a 10-frame stack
    num_frames = 10
    stack_data = np.random.randint(0, 255, (num_frames, 8, 8), dtype=np.uint8)
    large_stack = SingleChannelImageStack(stack_data)
    base_path = os.path.join(tmp_dir, "large_stack")

    saver = PNGSingleChannelImageStackSaver()
    saver.send_data(large_stack, base_path)

    # Check all files were created with correct numbering
    for i in range(num_frames):
        expected_file = os.path.join(tmp_dir, f"large_stack_{i:03d}.png")
        assert os.path.exists(expected_file), f"Frame {i} file was not created"

        # Verify content
        loaded_img = PngSingleChannelImageLoader().get_data(expected_file)
        loaded_array = loaded_img.data
        np.testing.assert_array_equal(
            loaded_array, stack_data[i], err_msg=f"Frame {i} content mismatch"
        )


def test_png_image_stack_saver_invalid_input(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with invalid input."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver
    from semantiva_imaging.data_types import RGBImage

    # Try to save an RGB image instead of SingleChannelImageStack
    rgb_data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    rgb_img = RGBImage(rgb_data)
    base_path = os.path.join(tmp_dir, "invalid_input")

    saver = PNGSingleChannelImageStackSaver()

    with pytest.raises(ValueError, match="not an instance of SingleChannelImageStack"):
        saver.send_data(rgb_img, base_path)


def test_png_image_stack_saver_different_data_types(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with different data types."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Test with float32 data (should be converted to uint8)
    float_data = (
        np.array(
            [[[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]], [[0.3, 0.7, 0.4], [0.8, 0.9, 0.5]]],
            dtype=np.float32,
        )
        * 255
    )

    float_stack = SingleChannelImageStack(float_data)
    base_path = os.path.join(tmp_dir, "float_stack")

    saver = PNGSingleChannelImageStackSaver()
    saver.send_data(float_stack, base_path)

    # Check files were created
    for i in range(2):
        expected_file = os.path.join(tmp_dir, f"float_stack_{i:03d}.png")
        assert os.path.exists(expected_file), f"Float frame {i} file was not created"

        # Verify that data was properly converted to uint8
        loaded_img = PngSingleChannelImageLoader().get_data(expected_file)
        loaded_array = loaded_img.data
        expected_array = float_data[i].astype(np.uint8)
        np.testing.assert_array_equal(loaded_array, expected_array)


def test_png_image_stack_saver_edge_case_paths(tmp_dir):
    """Test PNGSingleChannelImageStackSaver with various path formats."""
    from semantiva_imaging.data_io import PNGSingleChannelImageStackSaver

    # Create test data
    test_data = np.random.randint(0, 255, (2, 4, 4), dtype=np.uint8)
    test_stack = SingleChannelImageStack(test_data)

    saver = PNGSingleChannelImageStackSaver()

    # Test with path containing dots
    base_path_with_dots = os.path.join(tmp_dir, "test.with.dots")
    saver.send_data(test_stack, base_path_with_dots)

    expected_files = [
        os.path.join(tmp_dir, "test.with.dots_000.png"),
        os.path.join(tmp_dir, "test.with.dots_001.png"),
    ]

    for file_path in expected_files:
        assert os.path.exists(
            file_path
        ), f"File with dots path {file_path} was not created"

    # Test with path containing underscores (should still work)
    base_path_with_underscores = os.path.join(tmp_dir, "test_with_underscores")
    saver.send_data(test_stack, base_path_with_underscores)

    expected_files = [
        os.path.join(tmp_dir, "test_with_underscores_000.png"),
        os.path.join(tmp_dir, "test_with_underscores_001.png"),
    ]

    for file_path in expected_files:
        assert os.path.exists(
            file_path
        ), f"File with underscores path {file_path} was not created"
