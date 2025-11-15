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

"""Tests for the local safe evaluator used in ParametricPlotGenerator."""

import numpy as np
import pytest

from semantiva_imaging.data_io.parametric_plotter import (
    ImagingExpressionEvaluator,
    ParametricPlotGenerator,
)
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure


def test_numpy_math_functions_allowed_vectorized():
    """Common NumPy-like math functions should be allowed and vectorized."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 2, 16)

    for expr in ["sin(x)", "cos(x)", "tanh(x)", "sqrt(x*x)", "exp(x)", "log10(x+1)"]:
        y = ev.evaluate_y(expr, x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_constants_pi_e_are_available():
    """Constants pi and e should be available as variables."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 1, 8)
    y = ev.evaluate_y("sin(pi*x) + e - e", x)
    y_expected = np.sin(np.pi * x)
    np.testing.assert_allclose(y, y_expected, rtol=1e-6, atol=1e-6)


def test_arctan2_and_clip_work():
    """Functions with multiple args like arctan2 and clip should work."""
    ev = ImagingExpressionEvaluator()
    x = np.array([-1.0, 0.0, 1.0], dtype=float)

    y_clip = ev.evaluate_y("clip(x, 0, 1)", x)
    assert y_clip.min() >= 0.0 and y_clip.max() <= 1.0

    y_atan2 = ev.evaluate_y("arctan2(x, x)", x)
    # For positive x, atan2(x, x) ≈ pi/4; for negative x, ≈ -3pi/4; for 0, 0
    assert pytest.approx(y_atan2[1], rel=1e-6, abs=1e-6) == 0.0
    assert y_atan2[2] == pytest.approx(np.pi / 4, rel=1e-6, abs=1e-6)


def test_reject_attribute_access_np_sin():
    """Attribute access like np.sin(x) must be rejected."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ev.evaluate_y("np.sin(x)", x)


def test_reject_disallowed_builtin_sum():
    """Non-whitelisted built-ins like sum() are rejected."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ev.evaluate_y("sum(x)", x)


def test_reject_subscript_and_lambda():
    """Subscripts and lambda expressions are not allowed in safe eval."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ev.evaluate_y("x[0]", x)
    with pytest.raises(ValueError, match="Invalid y_expression"):
        ev.evaluate_y("(lambda t: t)(x)", x)


def test_plotter_integration_with_numpy_math():
    """End-to-end: plotter should accept NumPy-like math safely."""
    result = ParametricPlotGenerator.get_data(
        x_range={"lo": 0, "hi": 2 * np.pi, "steps": 32},
        y_expression="sin(x) + 0.5*cos(2*x)",
        x_label="x",
        y_label="y",
        title="sin+cos",
    )
    assert isinstance(result, MatplotlibFigure)


def test_eval_failure_on_nonvectorized_int():
    """Calling int(x) on arrays fails and should be wrapped as evaluation failure."""
    ev = ImagingExpressionEvaluator()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Failed to evaluate y_expression"):
        # int(np.array([...])) raises a TypeError internally
        ev.evaluate_y("int(x)", x)
