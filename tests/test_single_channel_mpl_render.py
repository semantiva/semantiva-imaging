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

"""Tests for SingleChannelImage → MatplotlibFigure processors."""

import numpy as np
import matplotlib.pyplot as plt

from semantiva_imaging.data_types import SingleChannelImage, SingleChannelImageStack
from semantiva_imaging.data_types.mpl_figure import (
    MatplotlibFigure,
    MatplotlibFigureCollection,
)
from semantiva_imaging.processing import (
    SingleChannelImageToMatplotlibFigure,
    SingleChannelImageStackToMatplotlibFigureCollection,
    FigureCollectionToRGBAStack,
)


def _make_test_image(shape=(32, 32)):
    """Create a test SingleChannelImage with random data."""
    data = np.random.rand(*shape).astype("float32") * 100
    return SingleChannelImage(data)


def _make_test_stack(n=3, shape=(32, 32)):
    """Create a test SingleChannelImageStack with random data."""
    frames = [np.random.rand(*shape).astype("float32") * 100 for _ in range(n)]
    return SingleChannelImageStack(np.stack(frames))


def test_single_image_to_figure_types():
    """Test that SingleChannelImageToMatplotlibFigure returns correct type."""
    img = _make_test_image()
    op = SingleChannelImageToMatplotlibFigure()
    result = op.process(img)

    assert isinstance(result, MatplotlibFigure)
    assert result.data is not None
    assert hasattr(result.data, "axes")
    assert len(result.data.axes) == 1

    # Clean up
    plt.close(result.data)


def test_single_image_to_figure_with_colorbar():
    """Test rendering with colorbar enabled."""
    img = _make_test_image()
    op = SingleChannelImageToMatplotlibFigure()
    result = op.process(img, colorbar=True, title="Test", cmap="viridis")

    assert isinstance(result, MatplotlibFigure)
    # Should have axes and colorbar
    assert len(result.data.axes) >= 1

    plt.close(result.data)


def test_single_image_to_figure_log_scale():
    """Test rendering with logarithmic scale."""
    img = _make_test_image()
    op = SingleChannelImageToMatplotlibFigure()
    result = op.process(img, log_scale=True, cmap="hot")

    assert isinstance(result, MatplotlibFigure)
    ax = result.data.axes[0]
    images = [obj for obj in ax.get_children() if hasattr(obj, "get_array")]
    assert len(images) > 0
    # Check that a normalization was applied
    assert images[0].norm is not None

    plt.close(result.data)


def test_single_image_to_figure_labels():
    """Test rendering with custom labels and title."""
    img = _make_test_image()
    op = SingleChannelImageToMatplotlibFigure()
    result = op.process(
        img, title="Test Title", xlabel="X Position", ylabel="Y Position"
    )

    assert isinstance(result, MatplotlibFigure)
    ax = result.data.axes[0]
    assert ax.get_title() == "Test Title"
    assert ax.get_xlabel() == "X Position"
    assert ax.get_ylabel() == "Y Position"

    plt.close(result.data)


def test_single_image_to_figure_different_cmaps():
    """Test rendering with different colormaps."""
    img = _make_test_image()
    op = SingleChannelImageToMatplotlibFigure()

    for cmap in ["hot", "viridis", "inferno", "plasma"]:
        result = op.process(img, cmap=cmap)
        assert isinstance(result, MatplotlibFigure)
        plt.close(result.data)


def test_stack_to_collection_types_and_length():
    """Test that stack processor returns collection with correct length."""
    stack = _make_test_stack(n=5)
    op = SingleChannelImageStackToMatplotlibFigureCollection()
    result = op.process(stack, title="Stack")

    assert isinstance(result, MatplotlibFigureCollection)
    assert len(result) == 5

    for item in result:
        assert isinstance(item, MatplotlibFigure)
        assert item.data is not None

    # Clean up
    for item in result:
        plt.close(item.data)


def test_stack_to_collection_empty_stack():
    """Test handling of empty stack."""
    # Create an empty stack
    empty_stack = SingleChannelImageStack(np.empty((0, 32, 32), dtype="float32"))
    op = SingleChannelImageStackToMatplotlibFigureCollection()
    result = op.process(empty_stack)

    assert isinstance(result, MatplotlibFigureCollection)
    assert len(result) == 0


def test_stack_to_collection_parameters_propagation():
    """Test that visualization parameters are applied to all frames."""
    stack = _make_test_stack(n=3)
    op = SingleChannelImageStackToMatplotlibFigureCollection()
    result = op.process(
        stack,
        title="Propagated Title",
        colorbar=True,
        cmap="plasma",
        xlabel="X",
        ylabel="Y",
    )

    assert len(result) == 3

    # Check that each figure has the same settings
    for item in result:
        ax = item.data.axes[0]
        assert ax.get_title() == "Propagated Title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    # Clean up
    for item in result:
        plt.close(item.data)


def test_pipeline_to_rgba_stack_shape():
    """Test the full pipeline from stack to RGBA."""
    stack = _make_test_stack(n=4, shape=(40, 20))

    # Stack → FigureCollection
    op_coll = SingleChannelImageStackToMatplotlibFigureCollection()
    coll = op_coll.process(stack, cmap="hot")

    # FigureCollection → RGBAImageStack
    op_rgba = FigureCollectionToRGBAStack()
    rgba_stack = op_rgba.process(coll, size_px=(200, 100), dpi=100, close_after=True)

    # Verify shape
    data = rgba_stack.data
    assert data.shape == (4, 100, 200, 4)  # (N, H, W, 4)
    assert data.dtype == np.uint8

    # Figures should already be closed by close_after=True


def test_pipeline_rgba_stack_consistency():
    """Test that all frames in RGBA stack have the same shape."""
    stack = _make_test_stack(n=10, shape=(50, 50))

    op_coll = SingleChannelImageStackToMatplotlibFigureCollection()
    coll = op_coll.process(stack)

    op_rgba = FigureCollectionToRGBAStack()
    rgba_stack = op_rgba.process(coll, size_px=(256, 256), dpi=80, close_after=True)

    data = rgba_stack.data
    assert data.shape[0] == 10  # 10 frames
    assert data.shape[1] == 256  # Height
    assert data.shape[2] == 256  # Width
    assert data.shape[3] == 4  # RGBA channels


def test_single_image_input_output_types():
    """Test that processor declares correct input/output types."""
    op = SingleChannelImageToMatplotlibFigure()
    assert op.input_data_type() == SingleChannelImage
    assert op.output_data_type() == MatplotlibFigure


def test_stack_input_output_types():
    """Test that stack processor declares correct input/output types."""
    op = SingleChannelImageStackToMatplotlibFigureCollection()
    assert op.input_data_type() == SingleChannelImageStack
    assert op.output_data_type() == MatplotlibFigureCollection


def test_stack_to_collection_various_sizes():
    """Test stack processing with different stack sizes."""
    for n in [1, 2, 5, 10]:
        stack = _make_test_stack(n=n)
        op = SingleChannelImageStackToMatplotlibFigureCollection()
        result = op.process(stack)

        assert len(result) == n

        # Clean up
        for item in result:
            plt.close(item.data)


def test_figure_has_image_data():
    """Test that generated figure contains actual image data."""
    img = _make_test_image(shape=(20, 30))
    op = SingleChannelImageToMatplotlibFigure()
    result = op.process(img)

    ax = result.data.axes[0]
    images = [obj for obj in ax.get_children() if hasattr(obj, "get_array")]

    assert len(images) > 0
    image_data = images[0].get_array()
    assert image_data.shape == (20, 30)

    plt.close(result.data)
