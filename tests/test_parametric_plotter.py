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

"""Tests for ParametricPlotGenerator processor with x_range + y_expression API."""

import pytest
from semantiva_imaging.data_io import ParametricPlotGenerator
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure


def test_simple_linear_plot():
    """Test generating a simple linear plot using x_range."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0.0, "hi": 10.0, "steps": 11},
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
        x_range={"lo": -2, "hi": 2, "steps": 100},
        y_expression="x**2",
        x_label="x",
        y_label="y = x²",
        title="Quadratic Function",
    )

    assert isinstance(result, MatplotlibFigure)


def test_parametric_expression():
    """Test parametric expression like those used in Gaussian beam examples."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": -1, "hi": 2, "steps": 50},
        y_expression="50 + 5*x + 5*x**2",
        x_label="t",
        y_label="y(t)",
        title="Parametric Expression",
    )

    assert isinstance(result, MatplotlibFigure)


def test_custom_styling():
    """Test plot with custom styling parameters."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 6.28, "steps": 100},
        y_expression="x",
        x_label="x",
        y_label="y",
        title="Styled Plot",
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
        x_range={"lo": 0, "hi": 1, "steps": 10},
        y_expression="x",
        grid=False,
    )

    assert isinstance(result, MatplotlibFigure)


def test_empty_title():
    """Test plot with no title."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 1, "steps": 10},
        y_expression="x",
        title="",
    )

    assert isinstance(result, MatplotlibFigure)


def test_constant_expression():
    """Test with a constant y expression."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 10, "steps": 20},
        y_expression="20 + 0*x",
        x_label="t",
        y_label="constant",
        title="Constant Function",
    )

    assert isinstance(result, MatplotlibFigure)


def test_invalid_y_expression():
    """Test that invalid y expression raises error."""
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 1, "steps": 10},
            y_expression="undefined_variable * 2",
        )


def test_security_reject_dangerous_expression():
    """Test that dangerous expressions are rejected by safe evaluator."""
    with pytest.raises(ValueError):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 1, "steps": 10},
            y_expression="__import__('os').system('ls')",
        )


def test_output_type():
    """Test that processor has correct output type."""
    assert ParametricPlotGenerator.output_data_type() == MatplotlibFigure


def test_missing_x_range_keys():
    """Test that error is raised when x_range is missing required keys."""
    with pytest.raises(ValueError, match="x_range must define 'lo', 'hi', 'steps'"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 10},  # missing 'steps'
            y_expression="2*x + 1",
        )


def test_invalid_steps():
    """Test that error is raised when steps <= 1."""
    with pytest.raises(ValueError, match="x_range.steps must be > 1"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 10, "steps": 1},
            y_expression="x",
        )


def test_zero_steps():
    """Test that error is raised when steps is zero."""
    with pytest.raises(ValueError, match="x_range.steps must be > 1"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 10, "steps": 0},
            y_expression="x",
        )


def test_negative_steps():
    """Test that error is raised when steps is negative."""
    with pytest.raises(ValueError, match="x_range.steps must be > 1"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 10, "steps": -5},
            y_expression="x",
        )


def test_non_numeric_x_range_values():
    """Test that error is raised when x_range values are not numeric."""
    with pytest.raises(ValueError, match="x_range values must be numeric"):
        ParametricPlotGenerator.get_data(
            x_range={"lo": "zero", "hi": "ten", "steps": 10},
            y_expression="x",
        )


def test_various_line_styles():
    """Test different matplotlib line styles."""
    line_styles = ["b-o", "r-s", "g-^", "m-d", "c-v"]

    for style in line_styles:
        result = ParametricPlotGenerator.get_data(
            x_range={"lo": 0, "hi": 5, "steps": 6},
            y_expression="x**2",
            line_style=style,
        )
        assert isinstance(result, MatplotlibFigure)


def test_complex_polynomial():
    """Test with a more complex polynomial expression."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": -2, "hi": 2, "steps": 100},
        y_expression="x**3 - 2*x**2 + x - 1",
        x_label="x",
        y_label="f(x)",
        title="Cubic Polynomial",
    )

    assert isinstance(result, MatplotlibFigure)


def test_expression_with_parentheses():
    """Test expression evaluation with parentheses."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 10, "steps": 20},
        y_expression="(x + 1) * (x - 2)",
        x_label="x",
        y_label="y",
    )

    assert isinstance(result, MatplotlibFigure)


def test_large_range():
    """Test with a large number of steps."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 100, "steps": 1000},
        y_expression="x**2",
        x_label="x",
        y_label="y",
    )

    assert isinstance(result, MatplotlibFigure)


def test_float_steps_conversion():
    """Test that float steps value is converted to int."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 10, "steps": 11.0},  # float instead of int
        y_expression="x",
    )

    assert isinstance(result, MatplotlibFigure)
