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

"""Safe-eval tests for :class:`ExpressionEnv` in ``parametric_base``.

These tests ensure that the shared expression evaluation environment used by
parametric line and surface generators:

* supports common NumPy-style math in a vectorized way
* exposes math constants ``pi`` and ``e``
* rejects disallowed syntax such as attribute access, subscripts, and lambdas
* restricts function calls to a whitelisted set of math helpers
* produces clear ``ValueError`` messages for invalid expressions
"""

from __future__ import annotations

import numpy as np
import pytest

from semantiva_imaging.data_io.parametric_base import ExpressionEnv


def test_numpy_math_functions_allowed_vectorized() -> None:
    """Common NumPy-like math functions should be allowed and vectorized."""
    env = ExpressionEnv()
    x = np.linspace(0, 2, 16)

    for expr in [
        "sin(x)",
        "cos(x)",
        "tanh(x)",
        "sqrt(x*x)",
        "exp(x)",
        "log10(x+1)",
    ]:
        y = env.evaluate(expr, x=x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_constants_pi_e_are_available() -> None:
    """Constants pi and e should be available as variables."""
    env = ExpressionEnv()
    x = np.linspace(0, 1, 8)
    y = env.evaluate("sin(pi*x) + e - e", x=x)
    y_expected = np.sin(np.pi * x)
    np.testing.assert_allclose(y, y_expected, rtol=1e-6, atol=1e-6)


def test_arctan2_and_clip_work() -> None:
    """Functions with multiple args like arctan2 and clip should work."""
    env = ExpressionEnv()
    x = np.array([-1.0, 0.0, 1.0], dtype=float)

    y_clip = env.evaluate("clip(x, 0, 1)", x=x)
    assert y_clip.min() >= 0.0 and y_clip.max() <= 1.0

    y_atan2 = env.evaluate("arctan2(x, x)", x=x)
    # For positive x, atan2(x, x) ≈ pi/4; for negative x, ≈ -3pi/4; for 0, 0
    assert pytest.approx(y_atan2[1], rel=1e-6, abs=1e-6) == 0.0
    assert y_atan2[2] == pytest.approx(np.pi / 4, rel=1e-6, abs=1e-6)


def test_reject_attribute_access_np_sin() -> None:
    """Attribute access like np.sin(x) must be rejected."""
    env = ExpressionEnv()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Only calls to whitelisted math functions"):
        env.evaluate("np.sin(x)", x=x)


def test_reject_disallowed_builtin_sum() -> None:
    """Non-whitelisted built-ins like sum() are rejected."""
    env = ExpressionEnv()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Only calls to whitelisted math functions"):
        env.evaluate("sum(x)", x=x)


def test_reject_subscript_and_lambda() -> None:
    """Subscripts and lambda expressions are not allowed in safe eval."""
    env = ExpressionEnv()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Disallowed syntax: Subscript"):
        env.evaluate("x[0]", x=x)
    with pytest.raises(ValueError, match="Only calls to whitelisted math functions"):
        env.evaluate("(lambda t: t)(x)", x=x)


def test_eval_failure_on_nonvectorized_int() -> None:
    """Calling int(x) on arrays fails and should be wrapped as evaluation failure."""
    env = ExpressionEnv()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match="Failed to evaluate expression"):
        # int(np.array([...])) raises a TypeError internally; ExpressionEnv
        # should translate this into a ValueError with a clear message.
        env.evaluate("int(x)", x=x)
