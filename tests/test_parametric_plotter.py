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

"""Tests for ParametricPlotGenerator processor."""

import pytest
import numpy as np
from semantiva_imaging.data_io import ParametricPlotGenerator
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure


def test_simple_linear_plot():
    """Test generating a simple linear plot."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(0, 10, 11),
        y_expression="2*x + 1",
        x_label="x",
        y_label="y",
        title="Linear Function",
    )

    assert isinstance(result, MatplotlibFigure)
    assert "MatplotlibFigure" in str(result)
    assert "axes=1" in str(result)


def test_quadratic_plot():
    """Test generating a quadratic plot."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(-2, 2, 100),
        y_expression="x**2",
        x_label="x",
        y_label="y = x²",
        title="Quadratic Function",
    )

    assert isinstance(result, MatplotlibFigure)


def test_parametric_with_t_variable():
    """Test using t as the variable name (common for parametric plots)."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(-1, 2, 50),
        y_expression="50 + 5*t + 5*t**2",
        x_label="t",
        y_label="y(t)",
        title="Parametric with t",
    )

    assert isinstance(result, MatplotlibFigure)


def test_custom_styling():
    """Test plot with custom styling parameters."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(0, 2 * np.pi, 100),
        y_expression="x",  # Simple expression since x already computed
        x_label="x",
        y_label="sin(x)",
        title="Sine Wave",
        figure_size=(10.0, 4.0),
        line_style="r--",
        line_width=3.0,
        marker_size=0,
        grid=True,
        grid_alpha=0.5,
    )

    assert isinstance(result, MatplotlibFigure)
    fig = result.data
    assert fig.get_figwidth() == 10.0
    assert fig.get_figheight() == 4.0


def test_no_grid():
    """Test plot without grid."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(0, 1, 10),
        y_expression="x",
        grid=False,
    )

    assert isinstance(result, MatplotlibFigure)


def test_empty_title():
    """Test plot with no title."""
    result = ParametricPlotGenerator.get_data(
        x_values=np.linspace(0, 1, 10),
        y_expression="x",
        title="",
    )

    assert isinstance(result, MatplotlibFigure)


def test_list_input():
    """Test with list input instead of numpy array."""
    result = ParametricPlotGenerator.get_data(
        x_values=[0, 1, 2, 3, 4, 5],
        y_expression="2*x + 1",
        x_label="x",
        y_label="y",
    )

    assert isinstance(result, MatplotlibFigure)


def test_invalid_y_expression():
    """Test that invalid y expression raises error."""
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ParametricPlotGenerator.get_data(
            x_values=np.linspace(0, 1, 10),
            y_expression="undefined_variable * 2",
        )


def test_security_reject_dangerous_expression():
    """Test that dangerous expressions are rejected."""
    with pytest.raises(ValueError):
        ParametricPlotGenerator.get_data(
            x_values=[1, 2, 3],
            y_expression="__import__('os').system('ls')",
        )


def test_output_type():
    """Test that processor has correct output type."""
    assert ParametricPlotGenerator.output_data_type() == MatplotlibFigure


def test_t_parameter_direct():
    """Test using 't' parameter directly (equivalent to x_values)."""
    result = ParametricPlotGenerator.get_data(
        t=[0, 1, 2, 3, 4, 5],
        y_expression="2*t + 1",
        x_label="t",
        y_label="y",
    )

    assert isinstance(result, MatplotlibFigure)


def test_t_with_x_expression():
    """Test using 't' parameter with x_expression (run_space pattern)."""
    result = ParametricPlotGenerator.get_data(
        t=np.linspace(-1, 2, 50),
        x_expression="t",
        y_expression="50 + 5*t",
        x_label="t",
        y_label="y(t)",
        title="Using t and x_expression",
    )

    assert isinstance(result, MatplotlibFigure)


def test_t_with_complex_x_expression():
    """Test using 't' with a more complex x_expression."""
    result = ParametricPlotGenerator.get_data(
        t=np.linspace(0, 10, 20),
        x_expression="2*t",
        y_expression="x**2",
        x_label="x = 2t",
        y_label="y = x²",
    )

    assert isinstance(result, MatplotlibFigure)


def test_missing_required_parameters():
    """Test that error is raised when neither x_values nor t are provided."""
    with pytest.raises(ValueError, match="Must provide either 'x_values' or 't'"):
        ParametricPlotGenerator.get_data(
            y_expression="2*x + 1",
        )
