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

"""Parametric plotting data source for generating matplotlib figures.

.. deprecated:: 0.2.2
   `ParametricPlotGenerator` is now a compatibility shim over
   `ParametricLinePlotGenerator`. New code should use
   `ParametricLinePlotGenerator` directly for clearer semantics and
   the improved domain/scalars/expressions model.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Tuple
from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure
from .io import MatplotlibFigureDataSource
from .parametric_line import ParametricLinePlotGenerator


class ParametricPlotGenerator(MatplotlibFigureDataSource):
    """Generate parametric plots as MatplotlibFigure (compatibility shim).

    .. deprecated:: 0.2.2
       This class is maintained for backward compatibility and internally
       delegates to `ParametricLinePlotGenerator`. New pipelines should
       use `ParametricLinePlotGenerator` directly for access to the full
       domain/scalars/expressions model.

    This data source creates 1D parametric plots from an x-range specification
    and a mathematical expression, maintaining the original interface for
    existing pipelines.

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

    See Also
    --------
    ParametricLinePlotGenerator : Recommended replacement with domain/scalars/expressions model
    ParametricSurfacePlotGenerator : For 2D surface plots
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

        This method maintains backward compatibility by translating the original
        x_range/y_expression interface to the new domain/expressions model and
        delegating to ParametricLinePlotGenerator.

        Parameters
        ----------
        x_range : Mapping[str, Any]
            Range specification with keys 'lo', 'hi', 'steps'.
            Example: ``{lo: -1, hi: 2, steps: 50}``
        y_expression : str
            Mathematical expression in variable 'x' for computing y values.
            Examples: ``"2*x + 1"``, ``"x**2"``, ``"50 + 5*x"``
        x_label : str, default "x"
            X-axis label
        y_label : str, default "y"
            Y-axis label
        title : str, default ""
            Plot title
        figure_size : Tuple[float, float], default (8.0, 6.0)
            Figure dimensions (width, height) in inches
        line_style : str, default "b-o"
            Matplotlib line style (e.g., "b-o", "r-s", "g-^")
        line_width : float, default 2.0
            Line width in points
        marker_size : float, default 8.0
            Marker size in points
        grid : bool, default True
            Enable grid display
        grid_alpha : float, default 0.3
            Grid transparency (0.0 = invisible, 1.0 = opaque)

        Returns
        -------
        MatplotlibFigure
            The generated figure wrapped in MatplotlibFigure type

        Raises
        ------
        ValueError
            If x_range is missing required keys or y_expression is invalid

        Notes
        -----
        This method delegates to ParametricLinePlotGenerator for actual
        implementation. Consider migrating to ParametricLinePlotGenerator
        directly for access to the enhanced domain/scalars/expressions model.
        """
        # Translate old interface to new domain/expressions model
        domain = {"x": x_range}
        expressions = {"y": y_expression}

        # Delegate to ParametricLinePlotGenerator
        return (
            ParametricLinePlotGenerator._get_data(  # pylint: disable=protected-access
                domain=domain,
                expressions=expressions,
                scalars=None,
                x_label=x_label,
                y_label=y_label,
                title=title,
                figure_size=figure_size,
                line_style=line_style,
                line_width=line_width,
                marker_size=marker_size,
                grid=grid,
                grid_alpha=grid_alpha,
            )
        )
