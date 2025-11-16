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

"""Tests for ParametricLinePlotGenerator and ParametricSurfacePlotGenerator.

Tests cover:
- Domain parsing and validation
- Expression evaluation with scalars
- Shape validation
- Error handling for invalid configurations
- Security (expression environment restrictions)
"""

import pytest
from semantiva_imaging.data_io import (
    ParametricLinePlotGenerator,
    ParametricSurfacePlotGenerator,
)
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure


# =============================================================================
# ParametricLinePlotGenerator Tests
# =============================================================================


class TestParametricLinePlotGenerator:
    """Test suite for 1D parametric line plots."""

    def test_simple_linear(self):
        """Test simple linear plot: y = 2*x + 1."""
        result = ParametricLinePlotGenerator.get_data(
            domain={"x": {"lo": 0, "hi": 10, "steps": 11}},
            expressions={"y": "2*x + 1"},
            x_label="x",
            y_label="y",
            title="Linear Function",
        )

        assert isinstance(result, MatplotlibFigure)
        assert "MatplotlibFigure" in str(result)
        assert "axes=1" in str(result)

    def test_wave_with_scalars(self):
        """Test wave equation with scalar parameters: y = sin(k*x - w*t)."""
        result = ParametricLinePlotGenerator.get_data(
            domain={"x": {"lo": 0, "hi": 6.283185307179586, "steps": 100}},
            expressions={"y": "sin(k*x - w*t)"},
            scalars={"k": 1.0, "w": 1.0, "t": 0.0},
            x_label="x",
            y_label="sin(kx - ωt)",
            title="Traveling Wave at t=0",
        )

        assert isinstance(result, MatplotlibFigure)

    def test_math_constants_available(self):
        """Test that pi and e constants are available in expressions."""
        result = ParametricLinePlotGenerator.get_data(
            domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
            expressions={"y": "pi * x + e"},
        )

        assert isinstance(result, MatplotlibFigure)

    def test_numpy_functions_available(self):
        """Test that NumPy functions (sin, cos, exp, etc.) are available."""
        result = ParametricLinePlotGenerator.get_data(
            domain={"x": {"lo": -1, "hi": 1, "steps": 50}},
            expressions={"y": "exp(x) + sin(2*pi*x) + sqrt(abs(x))"},
        )

        assert isinstance(result, MatplotlibFigure)

    def test_custom_styling(self):
        """Test plot with custom styling parameters."""
        result = ParametricLinePlotGenerator.get_data(
            domain={"x": {"lo": 0, "hi": 10, "steps": 100}},
            expressions={"y": "x"},
            x_label="x",
            y_label="y",
            title="Styled Plot",
            figure_size=(12.0, 4.0),
            line_style="r--",
            line_width=3.0,
            marker_size=8.0,
            grid=True,
            grid_alpha=0.5,
        )

        assert isinstance(result, MatplotlibFigure)
        fig = result.data
        assert fig.get_figwidth() == 12.0
        assert fig.get_figheight() == 4.0

    def test_invalid_no_domain(self):
        """Test error when domain is empty."""
        with pytest.raises(ValueError, match="Domain must define at least one axis"):
            ParametricLinePlotGenerator.get_data(
                domain={},
                expressions={"y": "x"},
            )

    def test_invalid_multiple_axes(self):
        """Test error when domain has more than 1 axis."""
        with pytest.raises(ValueError, match="requires exactly 1 domain axis"):
            ParametricLinePlotGenerator.get_data(
                domain={
                    "x": {"lo": 0, "hi": 1, "steps": 10},
                    "y": {"lo": 0, "hi": 1, "steps": 10},
                },
                expressions={"y": "x"},
            )

    def test_invalid_missing_y_expression(self):
        """Test error when expressions['y'] is missing."""
        with pytest.raises(ValueError, match="requires expressions\\['y'\\]"):
            ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
                expressions={"z": "x"},  # Wrong key!
            )

    def test_invalid_expression_unknown_variable(self):
        """Test error when expression references unknown variable."""
        with pytest.raises(ValueError, match="Unknown variable 'foo'"):
            ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
                expressions={"y": "foo + x"},
            )

    def test_invalid_domain_steps_too_small(self):
        """Test error when domain steps < 2."""
        with pytest.raises(ValueError, match="steps must be >= 2"):
            ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 1}},
                expressions={"y": "x"},
            )

    def test_secure_eval_rejects_import(self):
        """Test that expression environment rejects __import__ attempts."""
        with pytest.raises(
            ValueError, match="Only calls to whitelisted math functions"
        ):
            ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
                expressions={"y": "__import__('os').system('echo bad')"},
            )

    def test_secure_eval_rejects_eval(self):
        """Test that expression environment rejects eval attempts."""
        with pytest.raises(
            ValueError, match="Only calls to whitelisted math functions"
        ):
            ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
                expressions={"y": "eval('1+1')"},
            )


# =============================================================================
# ParametricSurfacePlotGenerator Tests
# =============================================================================


class TestParametricSurfacePlotGenerator:
    """Test suite for 2D parametric surface plots."""

    def test_simple_paraboloid(self):
        """Test simple 2D function: z = x**2 + y**2."""
        result = ParametricSurfacePlotGenerator.get_data(
            domain={
                "x": {"lo": -2, "hi": 2, "steps": 50},
                "y": {"lo": -2, "hi": 2, "steps": 50},
            },
            expressions={"z": "x**2 + y**2"},
            x_label="x",
            y_label="y",
            title="Paraboloid",
        )

        assert isinstance(result, MatplotlibFigure)
        assert "MatplotlibFigure" in str(result)

    def test_standing_wave_with_scalars(self):
        """Test 2D standing wave: z = sin(kx*x) * sin(ky*y) * cos(w*t)."""
        result = ParametricSurfacePlotGenerator.get_data(
            domain={
                "x": {"lo": -3.14159, "hi": 3.14159, "steps": 80},
                "y": {"lo": -3.14159, "hi": 3.14159, "steps": 80},
            },
            expressions={"z": "sin(kx*x) * sin(ky*y) * cos(w*t)"},
            scalars={"kx": 2.0, "ky": 3.0, "w": 1.0, "t": 0.0},
            title="Standing Wave at t=0",
        )

        assert isinstance(result, MatplotlibFigure)

    def test_custom_colormap_and_range(self):
        """Test surface with custom colormap and vmin/vmax."""
        result = ParametricSurfacePlotGenerator.get_data(
            domain={
                "x": {"lo": -1, "hi": 1, "steps": 40},
                "y": {"lo": -1, "hi": 1, "steps": 40},
            },
            expressions={"z": "sin(pi*x) * cos(pi*y)"},
            cmap="RdBu_r",
            colorbar=True,
            vmin=-1.0,
            vmax=1.0,
        )

        assert isinstance(result, MatplotlibFigure)

    def test_figure_size_customization(self):
        """Test surface plot with custom figure size."""
        result = ParametricSurfacePlotGenerator.get_data(
            domain={
                "x": {"lo": 0, "hi": 1, "steps": 30},
                "y": {"lo": 0, "hi": 1, "steps": 30},
            },
            expressions={"z": "x + y"},
            figure_size=(10.0, 8.0),
        )

        assert isinstance(result, MatplotlibFigure)
        fig = result.data
        assert fig.get_figwidth() == 10.0
        assert fig.get_figheight() == 8.0

    def test_math_constants_available_2d(self):
        """Test that pi and e are available in 2D expressions."""
        result = ParametricSurfacePlotGenerator.get_data(
            domain={
                "x": {"lo": 0, "hi": 1, "steps": 20},
                "y": {"lo": 0, "hi": 1, "steps": 20},
            },
            expressions={"z": "pi * x + e * y"},
        )

        assert isinstance(result, MatplotlibFigure)

    def test_invalid_no_domain_2d(self):
        """Test error when domain is empty."""
        with pytest.raises(ValueError, match="Domain must define at least one axis"):
            ParametricSurfacePlotGenerator.get_data(
                domain={},
                expressions={"z": "x + y"},
            )

    def test_invalid_one_axis_only(self):
        """Test error when domain has only 1 axis (needs 2)."""
        with pytest.raises(ValueError, match="requires exactly 2 domain axes"):
            ParametricSurfacePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 1, "steps": 10}},
                expressions={"z": "x"},
            )

    def test_invalid_three_axes(self):
        """Test error when domain has 3 axes (needs 2)."""
        with pytest.raises(ValueError, match="requires exactly 2 domain axes"):
            ParametricSurfacePlotGenerator.get_data(
                domain={
                    "x": {"lo": 0, "hi": 1, "steps": 10},
                    "y": {"lo": 0, "hi": 1, "steps": 10},
                    "z": {"lo": 0, "hi": 1, "steps": 10},
                },
                expressions={"z": "x + y"},
            )

    def test_invalid_missing_z_expression(self):
        """Test error when expressions['z'] is missing."""
        with pytest.raises(ValueError, match="requires expressions\\['z'\\]"):
            ParametricSurfacePlotGenerator.get_data(
                domain={
                    "x": {"lo": 0, "hi": 1, "steps": 10},
                    "y": {"lo": 0, "hi": 1, "steps": 10},
                },
                expressions={"y": "x + y"},  # Wrong key!
            )

    def test_invalid_expression_unknown_variable_2d(self):
        """Test error when 2D expression references unknown variable."""
        with pytest.raises(ValueError, match="Unknown variable 'foo'"):
            ParametricSurfacePlotGenerator.get_data(
                domain={
                    "x": {"lo": 0, "hi": 1, "steps": 10},
                    "y": {"lo": 0, "hi": 1, "steps": 10},
                },
                expressions={"z": "foo + x + y"},
            )

    def test_secure_eval_rejects_import_2d(self):
        """Test that 2D expression environment rejects __import__ attempts."""
        with pytest.raises(
            ValueError, match="Only calls to whitelisted math functions"
        ):
            ParametricSurfacePlotGenerator.get_data(
                domain={
                    "x": {"lo": 0, "hi": 1, "steps": 10},
                    "y": {"lo": 0, "hi": 1, "steps": 10},
                },
                expressions={"z": "__import__('os')"},
            )


# =============================================================================
# Integration Tests (Pipeline-style usage)
# =============================================================================


class TestParametricIntegration:
    """Integration tests mimicking pipeline usage patterns."""

    def test_line_plot_parameter_sweep_pattern(self):
        """Test pattern used in wave animation: sweep scalar parameter."""
        # Simulate 3 time steps for a traveling wave
        results = []
        for t_val in [0.0, 1.0, 2.0]:
            result = ParametricLinePlotGenerator.get_data(
                domain={"x": {"lo": 0, "hi": 6.28, "steps": 100}},
                expressions={"y": "sin(k*x - w*t)"},
                scalars={"k": 1.0, "w": 1.0, "t": t_val},
                title=f"Wave at t={t_val:.1f}",
            )
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, MatplotlibFigure) for r in results)

    def test_surface_plot_parameter_sweep_pattern(self):
        """Test pattern used in surface wave animation: sweep scalar parameter."""
        # Simulate 3 time steps for a standing wave
        results = []
        for t_val in [0.0, 1.0, 2.0]:
            result = ParametricSurfacePlotGenerator.get_data(
                domain={
                    "x": {"lo": -3.14, "hi": 3.14, "steps": 60},
                    "y": {"lo": -3.14, "hi": 3.14, "steps": 60},
                },
                expressions={"z": "sin(kx*x) * sin(ky*y) * cos(w*t)"},
                scalars={"kx": 2.0, "ky": 3.0, "w": 1.0, "t": t_val},
                title=f"Standing Wave at t={t_val:.1f}",
                vmin=-1.0,
                vmax=1.0,
            )
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, MatplotlibFigure) for r in results)
