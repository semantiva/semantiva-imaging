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

"""Base engine for parametric plot generation.

This module provides the shared foundation for parametric plotting in
semantiva-imaging, implementing:

- Domain axis specification and sampling
- Scalar parameter handling
- Expression evaluation with imaging-safe math environment
- Grid generation for 1D and 2D parametric plots

The base engine is used by ParametricLinePlotGenerator (1D) and
ParametricSurfacePlotGenerator (2D) to provide a consistent, declarative
interface for generating matplotlib figures from mathematical expressions.
"""

from __future__ import annotations
import ast
from collections.abc import Mapping
from typing import Any, Callable
import numpy as np


class ExpressionEnv:
    """Safe expression evaluation environment for parametric plots.

    Provides a secure, restricted evaluation context for mathematical
    expressions used in parametric plots. Builds on Python's AST
    validation to prevent code injection while supporting common
    mathematical operations and NumPy functions.

    The environment exposes:
    - Mathematical constants: pi, e
    - NumPy functions: sin, cos, tan, exp, log, sqrt, etc.
    - Python built-ins: abs, min, max, round
    - Arithmetic and comparison operators

    Examples
    --------
    Basic expression evaluation:

    >>> env = ExpressionEnv()
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = env.evaluate("sin(x)", x=x)
    >>> y.shape == x.shape
    True

    With scalar parameters:

    >>> y = env.evaluate("sin(k*x - w*t)", x=x, k=2.0, w=1.0, t=0.5)

    Security (rejected expressions):

    >>> env.evaluate("__import__('os').system('echo bad')", x=x)
    Traceback (most recent call last):
        ...
    ValueError: Disallowed syntax: Call to non-whitelisted function
    """

    def __init__(self) -> None:
        """Initialize the expression evaluation environment."""
        # Built-in safe callables
        builtins: dict[str, Callable] = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "float": float,
            "int": int,
            "len": len,
        }

        # Whitelist safe NumPy math functions for vectorized evaluation
        numpy_funcs: dict[str, Callable] = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            "arctan2": np.arctan2,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "log2": np.log2,
            "sqrt": np.sqrt,
            "floor": np.floor,
            "ceil": np.ceil,
            "clip": np.clip,
            "deg2rad": np.deg2rad,
            "rad2deg": np.rad2deg,
            "power": np.power,
            "square": np.square,
            "sign": np.sign,
        }

        # Environment used during evaluation (no modules, just names → callables)
        self._env: dict[str, Callable] = {**builtins, **numpy_funcs}
        self._allowed_func_names: set[str] = set(self._env.keys())

        # Constants exposed as variables
        self._math_constants: dict[str, Any] = {
            "pi": float(np.pi),
            "e": float(np.e),
        }

    class _SafeVisitor(ast.NodeVisitor):
        """AST validator allowing simple math with whitelisted names only."""

        _ALLOWED_NODES = {
            ast.Expression,
            ast.Module,
            ast.Expr,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.IfExp,
            ast.Call,
            ast.Name,
            ast.Constant,
            ast.Tuple,
            ast.List,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        }

        def __init__(self, allowed_names: set[str], allowed_funcs: set[str]):
            self.allowed_names = allowed_names
            self.allowed_funcs = allowed_funcs

        def visit_Name(self, node: ast.Name) -> Any:
            if node.id not in self.allowed_names and node.id not in self.allowed_funcs:
                raise ValueError(
                    f"Unknown variable '{node.id}' in expression. "
                    f"Allowed variables: {sorted(self.allowed_names)}"
                )

        def visit_Call(self, node: ast.Call) -> Any:
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in self.allowed_funcs
            ):
                raise ValueError(
                    f"Only calls to whitelisted math functions are allowed. "
                    f"Available: {sorted(self.allowed_funcs)}"
                )
            # Validate positional arguments
            for arg in node.args:
                self.visit(arg)
            # Validate keyword arguments (security: prevent injection via kwargs)
            for keyword in node.keywords:
                self.visit(keyword.value)

        def generic_visit(self, node: ast.AST) -> Any:
            if type(node) not in self._ALLOWED_NODES:
                raise ValueError(
                    f"Disallowed syntax: {type(node).__name__}. "
                    f"Only basic mathematical expressions are allowed."
                )
            super().generic_visit(node)

    def _compile(self, expr: str, allowed_names: set[str]) -> Callable[..., Any]:
        """
        Compile expression to callable after AST validation.

        Parameters
        ----------
        expr : str
            Mathematical expression string
        allowed_names : set[str]
            Variable names that can appear in the expression

        Returns
        -------
        Callable
            Compiled function that takes keyword arguments

        Raises
        ------
        ValueError
            If expression has syntax errors or disallowed constructs
        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid expression '{expr}': {exc.msg} at position {exc.offset}"
            ) from exc

        self._SafeVisitor(allowed_names, self._allowed_func_names).visit(tree)
        code = compile(tree, filename="<expr>", mode="eval")

        def _fn(**kwargs: Any) -> Any:
            return eval(code, self._env, kwargs)  # pylint: disable=eval-used

        return _fn

    def evaluate(self, expr: str, **variables: Any) -> np.ndarray:
        """
        Evaluate mathematical expression with provided variables.

        Parameters
        ----------
        expr : str
            Mathematical expression string (e.g., "sin(k*x - w*t)", "x**2 + y**2")
        **variables : Any
            Variables to use in evaluation (axis arrays, scalar parameters)

        Returns
        -------
        np.ndarray
            Evaluated expression result

        Raises
        ------
        ValueError
            If expression is invalid, uses disallowed constructs, or
            references unknown variables

        Examples
        --------
        >>> env = ExpressionEnv()
        >>> x = np.linspace(0, 1, 10)
        >>> result = env.evaluate("2*x + 1", x=x)
        >>> result[0], result[-1]
        (1.0, 3.0)
        """
        allowed = set(variables.keys()) | set(self._math_constants.keys())
        try:
            fn = self._compile(expr, allowed_names=allowed)
            value = fn(**variables, **self._math_constants)
        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}") from e

        return np.asarray(value)


class DomainSpec:
    """Domain axis specification for parametric plots.

    Represents a single axis sampling specification with start, stop,
    and step count, similar to numpy.linspace but stored declaratively
    for YAML configuration.

    Parameters
    ----------
    lo : float
        Lower bound (inclusive)
    hi : float
        Upper bound (inclusive)
    steps : int
        Number of samples (must be >= 2)

    Raises
    ------
    ValueError
        If steps < 2 or bounds are invalid

    Examples
    --------
    >>> spec = DomainSpec(lo=0.0, hi=1.0, steps=11)
    >>> spec.to_array()
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    """

    def __init__(self, lo: float, hi: float, steps: int):
        if steps < 2:
            raise ValueError(f"Domain steps must be >= 2, got {steps}")
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Domain bounds must be finite: lo={lo}, hi={hi}")
        if lo >= hi:
            raise ValueError(f"Domain lo must be < hi: lo={lo}, hi={hi}")

        self.lo = float(lo)
        self.hi = float(hi)
        self.steps = int(steps)

    def to_array(self) -> np.ndarray:
        """Generate the sampled axis array."""
        return np.linspace(self.lo, self.hi, self.steps)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> DomainSpec:
        """
        Create DomainSpec from dictionary.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Dictionary with keys 'lo', 'hi', 'steps'

        Returns
        -------
        DomainSpec
            Parsed domain specification

        Raises
        ------
        ValueError
            If required keys are missing or values are invalid
        """
        try:
            lo = float(spec["lo"])
            hi = float(spec["hi"])
            steps = int(spec["steps"])
        except KeyError as e:
            raise ValueError(
                f"Domain spec must define 'lo', 'hi', 'steps': missing {e}"
            ) from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Domain spec values must be numeric: {e}") from e

        return cls(lo=lo, hi=hi, steps=steps)


class BaseParametricPlotGenerator:
    """
    Base engine for parametric plot generation.

    Provides shared functionality for 1D and 2D parametric plot generators:
    - Domain axis parsing and sampling
    - Scalar parameter handling
    - Expression evaluation with safe math environment
    - Grid generation for multi-dimensional domains

    This class is not used directly; instead, use:
    - ParametricLinePlotGenerator for 1D curves
    - ParametricSurfacePlotGenerator for 2D surfaces

    The base engine implements the domain/scalars/expressions model:

    domain:
      x: {lo: 0, hi: 10, steps: 100}
      y: {lo: -1, hi: 1, steps: 50}

    scalars:
      k: 1.0
      w: 2.0
      t: 0.0

    expressions:
      z: "sin(k*x) * cos(w*y - t)"

    Notes
    -----
    Subclasses must implement:
    - Axis count validation
    - Expression name requirements (e.g., 'y' for 1D, 'z' for 2D)
    - Matplotlib figure construction
    """

    def __init__(self):
        """Initialize the base parametric plot generator."""
        self._expr_env = ExpressionEnv()

    def _parse_domain(self, domain: Mapping[str, Any]) -> dict[str, DomainSpec]:
        """
        Parse domain specification from configuration.

        Parameters
        ----------
        domain : Mapping[str, Any]
            Domain configuration with axis names → range specs

        Returns
        -------
        dict[str, DomainSpec]
            Parsed domain specifications per axis

        Raises
        ------
        ValueError
            If domain is empty or specs are invalid
        """
        if not domain:
            raise ValueError("Domain must define at least one axis")

        parsed: dict[str, DomainSpec] = {}
        for axis_name, spec in domain.items():
            if not isinstance(axis_name, str):
                raise ValueError(f"Axis name must be string, got {type(axis_name)}")
            parsed[axis_name] = DomainSpec.from_dict(spec)

        return parsed

    def _build_axes_1d(
        self, domain_specs: dict[str, DomainSpec]
    ) -> dict[str, np.ndarray]:
        """
        Build 1D axis arrays from domain specs.

        Parameters
        ----------
        domain_specs : dict[str, DomainSpec]
            Parsed domain specifications

        Returns
        -------
        dict[str, np.ndarray]
            Axis name → 1D array mapping
        """
        return {name: spec.to_array() for name, spec in domain_specs.items()}

    def _build_grid_2d(
        self, domain_specs: dict[str, DomainSpec]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Build 2D grid arrays from domain specs.

        Parameters
        ----------
        domain_specs : dict[str, DomainSpec]
            Parsed domain specifications (must have exactly 2 axes)

        Returns
        -------
        axes_1d : dict[str, np.ndarray]
            Axis name → 1D array (for axis labeling)
        axes_grid : dict[str, np.ndarray]
            Axis name → 2D meshgrid array (for evaluation)

        Raises
        ------
        ValueError
            If domain doesn't have exactly 2 axes
        """
        if len(domain_specs) != 2:
            raise ValueError(
                f"Grid construction requires exactly 2 axes, got {len(domain_specs)}"
            )

        axes_1d = self._build_axes_1d(domain_specs)
        axis_names = list(axes_1d.keys())

        # Create meshgrid with 'xy' indexing (matrix/image convention)
        # For axes (x, y): rows vary in y, columns vary in x
        X, Y = np.meshgrid(
            axes_1d[axis_names[0]], axes_1d[axis_names[1]], indexing="xy"
        )

        axes_grid = {axis_names[0]: X, axis_names[1]: Y}

        return axes_1d, axes_grid

    def _evaluate_expression(
        self,
        expr: str,
        axes: dict[str, np.ndarray],
        scalars: dict[str, Any],
    ) -> np.ndarray:
        """
        Evaluate mathematical expression with axes and scalars.

        Parameters
        ----------
        expr : str
            Mathematical expression string
        axes : dict[str, np.ndarray]
            Axis arrays (1D or 2D depending on context)
        scalars : dict[str, Any]
            Scalar parameters

        Returns
        -------
        np.ndarray
            Evaluated expression result

        Raises
        ------
        ValueError
            If expression evaluation fails
        """
        variables = {**axes, **scalars}
        return self._expr_env.evaluate(expr, **variables)

    def _validate_expression_shape(
        self,
        result: np.ndarray,
        expected_shape: tuple[int, ...],
        expr_name: str,
        expr: str,
    ) -> None:
        """
        Validate that evaluated expression has expected shape.

        Parameters
        ----------
        result : np.ndarray
            Evaluated expression result
        expected_shape : tuple[int, ...]
            Expected array shape
        expr_name : str
            Name of expression (for error messages)
        expr : str
            Expression string (for error messages)

        Raises
        ------
        ValueError
            If shape doesn't match expected
        """
        if result.shape != expected_shape:
            raise ValueError(
                f"Expression '{expr_name}' = '{expr}' produced shape {result.shape}, "
                f"expected {expected_shape}"
            )
