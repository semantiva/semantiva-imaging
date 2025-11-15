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

"""Parametric plotting data source for generating matplotlib figures."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from semantiva.utils.safe_eval import ExpressionEvaluator, ExpressionError
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure
from .io import MatplotlibFigureDataSource


class ImagingExpressionEvaluator:
    """Safe expression evaluation for imaging parametric plots.

    This thin wrapper around Semantiva's core ExpressionEvaluator provides
    a future extension point for imaging-specific mathematical functions
    without requiring changes to the core framework.
    """

    def __init__(self) -> None:
        self._base = ExpressionEvaluator()
        # Future extension point for math environment (e.g., {"pi": np.pi, "sqrt": np.sqrt})
        self._math_env: dict[str, Any] = {}

    def evaluate_y(self, expr: str, x: np.ndarray) -> np.ndarray:
        """
        Evaluate y-expression in terms of variable x.

        Parameters
        ----------
        expr : str
            Mathematical expression string (e.g., "2*x + 1", "x**2")
        x : np.ndarray
            Array of x values to use in evaluation

        Returns
        -------
        np.ndarray
            Evaluated y values

        Raises
        ------
        ValueError
            If expression is invalid or evaluation fails
        """
        allowed = {"x", *self._math_env.keys()}
        try:
            fn = self._base.compile(expr, allowed_names=allowed)
            value = fn(x=x, **self._math_env)
        except ExpressionError as e:
            raise ValueError(f"Invalid y_expression '{expr}': {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to evaluate y_expression '{expr}': {e}") from e
        return np.asarray(value)


class ParametricPlotGenerator(MatplotlibFigureDataSource):
    """Generate synthetic parametric plots as MatplotlibFigure.

    This data source creates 2D parametric plots from a range specification
    and a mathematical expression, providing a pure, reproducible plotting
    primitive for Semantiva imaging pipelines.

    Examples
    --------
    Simple linear plot:

    .. code-block:: yaml

       - processor: ParametricPlotGenerator
         parameters:
           x_range: {lo: 0, hi: 10, steps: 50}
           y_expression: "2*x + 1"
           x_label: "x"
           y_label: "y"
           title: "Linear function"

    With parameter sweep:

    .. code-block:: yaml

       - processor: ParametricPlotGenerator
         parameters:
           x_range: {lo: -1, hi: 2, steps: 50}
         derive:
           parameter_sweep:
             parameters:
               y_expression: "equation"
               title: "title"
             variables:
               equation: ["2*x", "x**2", "x**3"]
               title: ["Linear", "Quadratic", "Cubic"]
             mode: by_position
             collection: MatplotlibFigureCollection
    """

    @classmethod
    def _get_data(
        cls,
        *,
        x_range: Mapping[str, Any],
        y_expression: str,
        x_label: str = "x",
        y_label: str = "y",
        title: str = "",
        figure_size: Tuple[float, float] = (8.0, 6.0),
        line_style: str = "b-o",
        line_width: float = 2.0,
        marker_size: float = 8.0,
        grid: bool = True,
        grid_alpha: float = 0.3,
    ) -> MatplotlibFigure:
        """
        Generate a parametric plot from a range specification and y expression.

        Parameters
        ----------
        x_range : Mapping[str, Any]
            Range specification with keys 'lo', 'hi', 'steps'.
            Interpreted identically to derive.parameter_sweep.variables ranges.
            Example: ``{lo: -1, hi: 2, steps: 50}``
        y_expression : str
            Mathematical expression in variable 'x' for computing y values.
            Uses safe evaluation (ExpressionEvaluator) to prevent code injection.
            Examples: ``"2*x + 1"``, ``"x**2"``, ``"50 + 5*x"``
        x_label : str, default "x"
            X-axis label.
        y_label : str, default "y"
            Y-axis label.
        title : str, default ""
            Plot title.
        figure_size : Tuple[float, float], default (8.0, 6.0)
            Figure dimensions (width, height) in inches.
        line_style : str, default "b-o"
            Matplotlib line style (e.g., "b-o", "r-s", "g-^").
        line_width : float, default 2.0
            Line width in points.
        marker_size : float, default 8.0
            Marker size in points.
        grid : bool, default True
            Enable grid display.
        grid_alpha : float, default 0.3
            Grid transparency (0.0 = invisible, 1.0 = opaque).

        Returns
        -------
        MatplotlibFigure
            The generated figure wrapped in MatplotlibFigure type.

        Raises
        ------
        ValueError
            If x_range is missing required keys, steps <= 1, or y_expression is invalid.

        Notes
        -----
        The generated figure uses the Agg backend (non-interactive). For interactive
        visualization, use separate viewer components or Semantiva Studio.

        The figure is NOT automatically closed. Use FigureToRGBAImage or
        FigureCollectionToRGBAStack with close_after=True to render and free resources,
        or rely on the automatic weakref finalizer in MatplotlibFigure.
        """
        # Decode x_range
        try:
            lo = float(x_range["lo"])
            hi = float(x_range["hi"])
            steps = int(x_range["steps"])
        except KeyError as e:
            raise ValueError(
                f"x_range must define 'lo', 'hi', 'steps': missing {e}"
            ) from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"x_range values must be numeric: {e}") from e

        if steps <= 1:
            raise ValueError("x_range.steps must be > 1")

        # Generate x values
        x = np.linspace(lo, hi, steps)

        # Evaluate y expression
        evaluator = ImagingExpressionEvaluator()
        y = evaluator.evaluate_y(y_expression, x=x)

        # Validate dimensions
        if x.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: x has shape {x.shape}, y has shape {y.shape}"
            )

        # Create figure
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111)

        # Plot data
        ax.plot(x, y, line_style, linewidth=line_width, markersize=marker_size)

        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # Configure grid
        if grid:
            ax.grid(True, alpha=grid_alpha)

        # Wrap in MatplotlibFigure (lifecycle managed by weakref finalizer)
        return MatplotlibFigure(fig)
