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

"""Tests for MatplotlibFigure data types."""

import matplotlib.pyplot as plt
import pytest
from semantiva_imaging.data_types.mpl_figure import (
    MatplotlibFigure,
    MatplotlibFigureCollection,
)


def test_matplotlib_figure_wrap_and_validate():
    """Test that a matplotlib figure can be wrapped and validated."""
    fig = plt.figure()
    wrapped = MatplotlibFigure(fig)
    assert "MatplotlibFigure(" in str(wrapped)
    assert "dpi=" in str(wrapped)
    assert "size_in=" in str(wrapped)
    assert "axes=" in str(wrapped)
    plt.close(fig)


def test_matplotlib_figure_invalid_data():
    """Test that non-figure data raises assertion error."""
    with pytest.raises(AssertionError, match="Expected matplotlib.figure.Figure"):
        MatplotlibFigure("not a figure")


def test_matplotlib_figure_with_axes():
    """Test figure with axes displays correct count."""
    fig = plt.figure()
    fig.add_subplot(111)
    wrapped = MatplotlibFigure(fig)
    assert "axes=1" in str(wrapped)
    plt.close(fig)


def test_collection_basic_methods():
    """Test basic collection methods: append, iter, len."""
    fig1 = plt.figure()
    fig2 = plt.figure()

    coll = MatplotlibFigureCollection([])
    assert len(coll) == 0

    coll.append(MatplotlibFigure(fig1))
    assert len(coll) == 1

    coll.append(MatplotlibFigure(fig2))
    assert len(coll) == 2

    items = list(iter(coll))
    assert len(items) == 2
    assert all(isinstance(item, MatplotlibFigure) for item in items)

    plt.close(fig1)
    plt.close(fig2)


def test_collection_initialization_with_list():
    """Test initializing collection with a list of figures."""
    fig1 = plt.figure()
    fig2 = plt.figure()

    coll = MatplotlibFigureCollection([MatplotlibFigure(fig1), MatplotlibFigure(fig2)])
    assert len(coll) == 2

    plt.close(fig1)
    plt.close(fig2)


def test_collection_invalid_item_type():
    """Test that non-MatplotlibFigure items raise assertion error."""
    with pytest.raises(AssertionError, match="expected MatplotlibFigure"):
        MatplotlibFigureCollection(["not a figure"])


def test_collection_mixed_invalid():
    """Test that mixed valid/invalid items raise assertion error."""
    fig = plt.figure()
    with pytest.raises(AssertionError, match="expected MatplotlibFigure"):
        MatplotlibFigureCollection([MatplotlibFigure(fig), "not a figure"])
    plt.close(fig)


def test_figure_repr_equals_str():
    """Test that __repr__ and __str__ are the same."""
    fig = plt.figure()
    wrapped = MatplotlibFigure(fig)
    assert repr(wrapped) == str(wrapped)
    plt.close(fig)
