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

"""Tests for figure render processors."""

import numpy as np
import matplotlib.pyplot as plt
import pytest
from semantiva_imaging.data_types.mpl_figure import (
    MatplotlibFigure,
    MatplotlibFigureCollection,
)
from semantiva_imaging.processing.figure_render import (
    FigureToRGBAImage,
    FigureCollectionToRGBAStack,
)


def _make_plot():
    """Helper to create a simple test plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_title("Test Plot")
    return fig


def test_single_render_to_rgba():
    """Test rendering a single figure to RGBA image."""
    fig = _make_plot()
    op = FigureToRGBAImage()
    out = op.process(
        MatplotlibFigure(fig), size_px=(320, 240), dpi=80, transparent=False
    )
    arr = out.data
    assert arr.dtype == np.uint8
    assert arr.shape == (240, 320, 4)
    plt.close(fig)


def test_single_render_different_sizes():
    """Test rendering at different sizes."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    # Test 640x480
    out1 = op.process(MatplotlibFigure(fig), size_px=(640, 480))
    assert out1.data.shape == (480, 640, 4)

    # Test 800x600
    out2 = op.process(MatplotlibFigure(fig), size_px=(800, 600))
    assert out2.data.shape == (600, 800, 4)

    plt.close(fig)


def test_single_render_different_dpi():
    """Test rendering at different DPI."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    out1 = op.process(MatplotlibFigure(fig), size_px=(400, 300), dpi=100)
    out2 = op.process(MatplotlibFigure(fig), size_px=(400, 300), dpi=150)

    # Both should have same pixel dimensions
    assert out1.data.shape == out2.data.shape == (300, 400, 4)

    plt.close(fig)


def test_single_render_transparent():
    """Test rendering with transparent background."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    # Non-transparent
    out_opaque = op.process(
        MatplotlibFigure(fig), size_px=(320, 240), transparent=False
    )

    # Transparent
    out_transparent = op.process(
        MatplotlibFigure(fig), size_px=(320, 240), transparent=True
    )

    # Both should be valid RGBA
    assert out_opaque.data.shape == (240, 320, 4)
    assert out_transparent.data.shape == (240, 320, 4)

    plt.close(fig)


def test_single_render_close_after():
    """Test that close_after parameter works."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    # Render with close_after=True
    out = op.process(MatplotlibFigure(fig), size_px=(320, 240), close_after=True)

    assert out.data.shape == (240, 320, 4)
    # Figure should be closed, but we can't easily test this directly


def test_collection_render_to_stack():
    """Test rendering a collection of figures to RGBA stack."""
    figs = [MatplotlibFigure(_make_plot()) for _ in range(3)]
    op = FigureCollectionToRGBAStack()
    out = op.process(MatplotlibFigureCollection(figs), size_px=(200, 100))
    stack = out.data
    assert stack.shape == (3, 100, 200, 4)
    assert stack.dtype == np.uint8

    # Close all figures
    for fig_wrapped in figs:
        plt.close(fig_wrapped.data)


def test_collection_render_empty():
    """Test rendering empty collection."""
    coll = MatplotlibFigureCollection([])
    op = FigureCollectionToRGBAStack()
    out = op.process(coll, size_px=(200, 100))
    stack = out.data
    # Should be empty stack with correct shape
    assert stack.shape[0] == 0


def test_collection_render_single_item():
    """Test rendering collection with single figure."""
    fig = _make_plot()
    coll = MatplotlibFigureCollection([MatplotlibFigure(fig)])
    op = FigureCollectionToRGBAStack()
    out = op.process(coll, size_px=(200, 150))
    stack = out.data
    assert stack.shape == (1, 150, 200, 4)
    plt.close(fig)


def test_collection_render_uniform_sizing():
    """Test that all figures in collection are resized uniformly."""
    # Create figures with different initial sizes
    fig1 = plt.figure(figsize=(4, 3))
    fig1.add_subplot(111).plot([0, 1], [0, 1])

    fig2 = plt.figure(figsize=(8, 6))
    fig2.add_subplot(111).plot([0, 1], [1, 0])

    coll = MatplotlibFigureCollection([MatplotlibFigure(fig1), MatplotlibFigure(fig2)])
    op = FigureCollectionToRGBAStack()

    out = op.process(coll, size_px=(400, 300))
    stack = out.data

    # Both should be resized to same dimensions
    assert stack.shape == (2, 300, 400, 4)

    plt.close(fig1)
    plt.close(fig2)


def test_render_input_output_types():
    """Test that processors have correct input/output types."""
    assert FigureToRGBAImage.input_data_type() == MatplotlibFigure
    assert FigureToRGBAImage.output_data_type().__name__ == "RGBAImage"

    assert FigureCollectionToRGBAStack.input_data_type() == MatplotlibFigureCollection
    assert FigureCollectionToRGBAStack.output_data_type().__name__ == "RGBAImageStack"


def test_render_invalid_size_px():
    """Test that invalid size_px raises assertion."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    with pytest.raises(AssertionError, match="size_px and dpi must be positive"):
        op.process(MatplotlibFigure(fig), size_px=(0, 240))

    with pytest.raises(AssertionError, match="size_px and dpi must be positive"):
        op.process(MatplotlibFigure(fig), size_px=(320, -240))

    plt.close(fig)


def test_render_invalid_dpi():
    """Test that invalid dpi raises assertion."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    with pytest.raises(AssertionError, match="size_px and dpi must be positive"):
        op.process(MatplotlibFigure(fig), size_px=(320, 240), dpi=0)

    with pytest.raises(AssertionError, match="size_px and dpi must be positive"):
        op.process(MatplotlibFigure(fig), size_px=(320, 240), dpi=-100)

    plt.close(fig)


def test_render_preserves_aspect_ratio():
    """Test that rendering respects specified pixel dimensions."""
    fig = _make_plot()
    op = FigureToRGBAImage()

    # Wide aspect ratio
    out_wide = op.process(MatplotlibFigure(fig), size_px=(800, 400))
    assert out_wide.data.shape == (400, 800, 4)

    # Tall aspect ratio
    out_tall = op.process(MatplotlibFigure(fig), size_px=(400, 800))
    assert out_tall.data.shape == (800, 400, 4)

    plt.close(fig)
