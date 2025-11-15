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

from typing import Tuple
import numpy as np
import matplotlib
import warnings

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from semantiva.utils.safe_eval import ExpressionEvaluator, ExpressionError
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure
from .io import MatplotlibFigureDataSource


class ParametricPlotGenerator(MatplotlibFigureDataSource):
    """Generate parametric plots as matplotlib figures from mathematical expressions."""

    @classmethod
    def _get_data(
        cls,
        *,
        x_values: list | tuple | None = None,
        t: list | tuple | np.ndarray | None = None,
        x_expression: str | None = None,
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
        show_via_qt: bool = False,
        ion: bool = False,
    ) -> MatplotlibFigure:
        """
        Generate a parametric plot from x values and a y expression.

        Parameters
        ----------
        x_values : list | tuple | np.ndarray | None
            X-axis values as raw data. Use with derive.parameter_sweep:

            .. code-block:: yaml

               variables:
                 t: { lo: -1, hi: 2, steps: 50 }
               parameters:
                 x_values: t

        t : list | tuple | np.ndarray | None
            Alternative parameter name for x-axis values. Used with x_expression
            to compute x from t via safe evaluation. Common in run_space:

            .. code-block:: yaml

               run_space:
                 context:
                   t: [generated from Range]
               parameters:
                 x_expression: "t"

        x_expression : str | None
            Safe mathematical expression to compute x-axis values from variable 't'.
            Available variable: t
            Examples: ``"t"``, ``"2*t"``, ``"t**2"``
            Used in run_space patterns where t comes from context.

        y_expression : str
            Expression to compute y-axis values from x.
            Available variables: x, t (alias for x)
            Example: ``"2*x + 1"``, ``"x**2"``, ``"50 + 5*t"``
        x_label : str
            X-axis label.
        y_label : str
            Y-axis label.
        title : str
            Plot title.
        figure_size : Tuple[float, float]
            Figure dimensions (width, height) in inches.
        line_style : str
            Matplotlib line style.
        line_width : float
            Line width.
        marker_size : float
            Marker size.
        grid : bool
            Enable grid display.
        grid_alpha : float
            Grid transparency.
        show_via_qt : bool
            Attempt to show via Qt (best-effort).
        ion : bool
            Enable interactive mode.

        Returns
        -------
        MatplotlibFigure
            The generated figure wrapped in MatplotlibFigure type.

        Raises
        ------
        ValueError
            If neither x_values nor (t + x_expression) are provided,
            or if expressions are invalid.
        """
        # Create safe evaluator
        evaluator = ExpressionEvaluator()

        # Determine x values from available parameters
        if x_values is not None:
            # Direct x_values provided (from parameter sweep or direct call)
            x = np.array(x_values)
        elif t is not None and x_expression is not None:
            # Compute x from t using x_expression (run_space pattern)
            t_array = np.array(t)
            try:
                x_fn = evaluator.compile(x_expression, allowed_names={"t"})
                x = x_fn(t=t_array)
                if not isinstance(x, np.ndarray):
                    x = np.array(x)
            except ExpressionError as e:
                raise ValueError(f"Invalid x_expression '{x_expression}': {e}")
            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate x_expression '{x_expression}': {e}"
                )
        elif t is not None:
            # t provided without x_expression, use t directly as x
            x = np.array(t)
        else:
            raise ValueError(
                "Must provide either 'x_values' or 't' (with optional 'x_expression')"
            )

        # Evaluate y expression safely (with x and t available)
        try:
            y_fn = evaluator.compile(y_expression, allowed_names={"x", "t"})
            y = y_fn(x=x, t=x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
        except ExpressionError as e:
            raise ValueError(f"Invalid y_expression '{y_expression}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to evaluate y_expression '{y_expression}': {e}")

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

        if ion:
            plt.ion()

        if show_via_qt:
            try:
                fig.show()
                if ion:
                    plt.pause(0.01)
            except Exception as e:
                warnings.warn(f"Interactive show failed: {e}")

        # Wrap in MatplotlibFigure
        return MatplotlibFigure(fig)
