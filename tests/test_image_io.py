import pytest
import os
import numpy as np
from semantiva_imaging.data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
)
from semantiva_imaging.data_io.loaders_savers import (
    ImageDataRandomGenerator,
    SingleChannelImageStackRandomGenerator,
    NpzSingleChannelImageLoader,
    NpzSingleChannelImageStackDataLoader,
    PngImageLoader,
    NpzImageDataSaver,
    NpzImageStackDataSaver,
    PngImageSaver,
)


@pytest.fixture
def sample_image_data():
    """
    Fixture to provide a sample SingleChannelImage using the dummy generator.
    """
    generator = ImageDataRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def sample_stack_data():
    """
    Fixture to provide a sample SingleChannelImageStack using the dummy generator.
    """
    generator = SingleChannelImageStackRandomGenerator()
    return generator.get_data((10, 256, 256))


@pytest.fixture
def tmp_test_dir(tmp_path):
    """
    Fixture to provide a temporary directory for creating test files.
    """
    return str(tmp_path)


def test_npz_image_loader(sample_image_data, tmp_test_dir):
    """
    Test loading SingleChannelImage from a dynamically created .npz file.
    """
    # Save the generated image to a .npz file
    file_path = os.path.join(tmp_test_dir, "image_data.npz")
    saver = NpzImageDataSaver()
    saver.send_data(sample_image_data, file_path)

    # Load the image back and verify
    loader = NpzSingleChannelImageLoader()
    image_data = loader.get_data(file_path)
    assert isinstance(image_data, SingleChannelImage)
    assert np.array_equal(image_data.data, sample_image_data.data)


def test_npz_stack_loader(sample_stack_data, tmp_test_dir):
    """
    Test loading SingleChannelImageStack from a dynamically created .npz file.
    """
    # Save the generated stack to a .npz file
    file_path = os.path.join(tmp_test_dir, "stack_data.npz")
    saver = NpzImageStackDataSaver()
    saver.send_data(sample_stack_data, file_path)

    # Load the stack back and verify
    loader = NpzSingleChannelImageStackDataLoader()
    stack_data = loader.get_data(file_path)
    assert isinstance(stack_data, SingleChannelImageStack)
    assert np.array_equal(stack_data.data, sample_stack_data.data)


def test_png_image_loader(sample_image_data, tmp_test_dir):
    """
    Test loading SingleChannelImage from a dynamically created .png file.
    """
    # Save the generated image to a .png file
    file_path = os.path.join(tmp_test_dir, "image_data.png")
    saver = PngImageSaver()
    saver.send_data(sample_image_data, file_path)

    # Load the image back and verify
    loader = PngImageLoader()
    image_data = loader.get_data(file_path)
    assert isinstance(image_data, SingleChannelImage)
    assert image_data.data.shape == sample_image_data.data.shape


def test_image_stack_iterator():
    """
    Tests the iterator of SingleChannelImageStack to ensure it correctly yields SingleChannelImage instances.
    """

    # Create a 3D numpy array (e.g., 5 images of size 10x10)
    num_images, height, width = 5, 10, 10
    stack_data = np.random.rand(num_images, height, width)

    # Create an ImageStackArrayDataType instance
    image_stack = SingleChannelImageStack(stack_data)

    assert image_stack.collection_base_type() == SingleChannelImage

    # Collect all elements from the iterator
    images = list(iter(image_stack))  # Equivalent to: [img for img in image_stack]

    # Validate the number of elements
    assert (
        len(images) == num_images
    ), "Iterator did not yield the expected number of elements"

    # Validate each element is an SingleChannelImage
    for img in images:
        assert isinstance(
            img, SingleChannelImage
        ), f"Iterator yielded an invalid type: {type(img)}"

        # Validate the internal NumPy array is 2D
        assert isinstance(
            img.data, np.ndarray
        ), "SingleChannelImage does not contain a NumPy array"
        assert (
            img.data.ndim == 2
        ), f"SingleChannelImage contains incorrect shape: {img.data.shape}"

    print("✅ test_image_stack_array_iterator passed!")
