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

"""1D parametric line plot generator for Semantiva Imaging.

This module provides ParametricLinePlotGenerator, a data source that creates
1D line plots from mathematical expressions of the form y(x, scalars).

The generator uses the domain/scalars/expressions model:
- domain.x: axis sampling specification
- scalars: additional parameters (k, w, t, etc.)
- expressions.y: mathematical expression for y values

This enables clean parametric animations via parameter sweeps on scalars.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Tuple
from matplotlib.figure import Figure

from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure
from .io import MatplotlibFigureDataSource
from .parametric_base import BaseParametricPlotGenerator


class ParametricLinePlotGenerator(MatplotlibFigureDataSource):
    """Generate 1D parametric line plots as MatplotlibFigure.

    This data source creates 1D line plots from a domain specification
    and mathematical expression, providing a declarative interface for
    plotting curves and animations.

    The generator requires:
    - **domain**: exactly one axis (typically 'x')
    - **expressions.y**: expression in axis and scalars
    - **scalars** (optional): additional parameters for the expression

    Examples
    --------
    Simple linear plot:

    .. code-block:: yaml

       - processor: ParametricLinePlotGenerator
         parameters:
           domain:
             x: {lo: 0, hi: 10, steps: 100}
           expressions:
             y: "2*x + 1"
           x_label: "x"
           y_label: "y"
           title: "Linear function"

    Wave animation with parameter sweep:

    .. code-block:: yaml

       - processor: ParametricLinePlotGenerator
         parameters:
           domain:
             x: {lo: -2*pi, hi: 2*pi, steps: 400}
           expressions:
             y: "sin(k*x - w*t)"
           scalars:
             k: 1.0
             w: 1.0
         derive:
           parameter_sweep:
             parameters:
               scalars.t: "t"
             variables:
               t: {lo: 0, hi: 6.283185307179586, steps: 24}
             mode: by_position
             collection: MatplotlibFigureCollection

    See Also
    --------
    ParametricSurfacePlotGenerator : For 2D scalar fields z(x, y, scalars)
    """

    def __init__(self):
        super().__init__()
        self._base = BaseParametricPlotGenerator()

    @classmethod
    def _get_data(
        cls,
        *,
        domain: Mapping[str, Any],
        expressions: Mapping[str, str],
        # Top-level scalar parameters for convenient sweeping (Option B)
        t: float | None = None,
        k: float | None = None,
        w: float | None = None,
        # Generic scalars mapping for advanced use cases
        scalars: Mapping[str, Any] | None = None,
        x_label: str = "x",
        y_label: str = "y",
        title: str = "",
        figure_size: Tuple[float, float] = (8.0, 6.0),
        line_style: str = "b-",
        line_width: float = 2.0,
        marker_size: float = 6.0,
        grid: bool = True,
        grid_alpha: float = 0.3,
    ) -> MatplotlibFigure:
        """
        Generate a 1D parametric line plot from domain and expression.

        Parameters
        ----------
        domain : Mapping[str, Any]
            Domain specification with exactly one axis. Each axis defines
            'lo', 'hi', 'steps'.
            Example: ``{x: {lo: 0, hi: 10, steps: 100}}``
        expressions : Mapping[str, str]
            Mathematical expressions. Must contain key 'y' with expression
            in terms of domain axis and scalars.
            Example: ``{y: "sin(k*x - w*t)"}``
        t : float, optional
            Time-like scalar parameter for expressions such as ``sin(k*x - w*t)``.
            Exposed as a top-level parameter to support direct sweeping via
            ``derive.parameter_sweep``.
        k : float, optional
            Wavenumber-like scalar parameter. Optional convenience parameter.
        w : float, optional
            Angular frequency-like scalar parameter. Optional convenience parameter.
        scalars : Mapping[str, Any], optional
            Additional scalar parameters available in expressions. Values here
            are merged with ``t``, ``k``, and ``w`` (top-level values override
            entries in ``scalars`` with the same key).
        x_label : str, default "x"
            X-axis label
        y_label : str, default "y"
            Y-axis label
        title : str, default ""
            Plot title
        figure_size : Tuple[float, float], default (8.0, 6.0)
            Figure dimensions (width, height) in inches
        line_style : str, default "b-"
            Matplotlib line style (e.g., "b-", "r--", "g-o")
        line_width : float, default 2.0
            Line width in points
        marker_size : float, default 6.0
            Marker size in points (if style includes markers)
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
            If domain doesn't have exactly 1 axis, expressions.y is missing,
            or expression evaluation fails

        Notes
        -----
        The figure is created without binding a GUI/backend-specific canvas.
        Rasterization processors attach an Agg canvas as needed for headless
        rendering. Use FigureToRGBAImage or FigureCollectionToRGBAStack with
        close_after=True to render and free resources, or rely on the
        automatic weakref finalizer in MatplotlibFigure.
        """
        # Create instance for base engine access
        instance = cls()

        # Parse and validate domain
        domain_specs = instance._base._parse_domain(
            domain
        )  # pylint: disable=protected-access
        if len(domain_specs) != 1:
            raise ValueError(
                f"ParametricLinePlotGenerator requires exactly 1 domain axis, "
                f"got {len(domain_specs)}: {list(domain_specs.keys())}"
            )

        # Build 1D axes
        axes_1d = instance._base._build_axes_1d(
            domain_specs
        )  # pylint: disable=protected-access
        axis_name = list(axes_1d.keys())[0]
        axis_array = axes_1d[axis_name]

        # Validate expressions
        if "y" not in expressions:
            raise ValueError(
                "ParametricLinePlotGenerator requires expressions['y']. "
                f"Got expressions keys: {list(expressions.keys())}"
            )

        y_expr = expressions["y"]

        # Prepare scalars (empty dict if None), then overlay top-level t/k/w
        scalar_params: dict[str, Any] = dict(scalars) if scalars else {}
        if t is not None:
            scalar_params["t"] = t
        if k is not None:
            scalar_params["k"] = k
        if w is not None:
            scalar_params["w"] = w

        # Evaluate y expression
        try:
            y_values = (
                instance._base._evaluate_expression(  # pylint: disable=protected-access
                    y_expr, axes_1d, scalar_params
                )
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to evaluate expressions['y'] = '{y_expr}': {e}"
            ) from e

        # Validate shape
        instance._base._validate_expression_shape(  # pylint: disable=protected-access
            y_values, axis_array.shape, "y", y_expr
        )

        # Create Figure instance directly (avoid pyplot state machine)
        fig = Figure(figsize=figure_size)
        ax = fig.add_subplot(111)

        # Plot data
        ax.plot(
            axis_array,
            y_values,
            line_style,
            linewidth=line_width,
            markersize=marker_size,
        )

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
