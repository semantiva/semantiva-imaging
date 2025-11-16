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

"""2D parametric surface plot generator for Semantiva Imaging.

This module provides ParametricSurfacePlotGenerator, a data source that creates
2D surface plots (heatmaps/colormaps) from mathematical expressions of the form
z(x, y, scalars).

The generator uses the domain/scalars/expressions model:
- domain.x, domain.y: axis sampling specifications
- scalars: additional parameters (t, kx, ky, w, etc.)
- expressions.z: mathematical expression for z values

This enables clean 2D surface animations via parameter sweeps on scalars.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Tuple, Literal
from matplotlib.figure import Figure

from semantiva_imaging.data_types.mpl_figure import MatplotlibFigure
from .io import MatplotlibFigureDataSource
from .parametric_base import BaseParametricPlotGenerator


class ParametricSurfacePlotGenerator(MatplotlibFigureDataSource):
    """Generate 2D parametric surface plots as MatplotlibFigure.

    This data source creates 2D scalar field visualizations (heatmaps) from
    domain specifications and mathematical expressions, providing a declarative
    interface for plotting surfaces and animations.

    The generator requires:
    - **domain**: exactly two axes (typically 'x' and 'y')
    - **expressions.z**: expression in axes and scalars
    - **scalars** (optional): additional parameters for the expression

    Examples
    --------
    Simple 2D function plot:

    .. code-block:: yaml

       - processor: ParametricSurfacePlotGenerator
         parameters:
           domain:
             x: {lo: -2, hi: 2, steps: 100}
             y: {lo: -2, hi: 2, steps: 100}
           expressions:
             z: "x**2 + y**2"
           x_label: "x"
           y_label: "y"
           title: "Paraboloid"
           cmap: "viridis"
           colorbar: true

    Standing wave animation with parameter sweep:

    .. code-block:: yaml

       - processor: ParametricSurfacePlotGenerator
         parameters:
           domain:
             x: {lo: -3.14159, hi: 3.14159, steps: 150}
             y: {lo: -3.14159, hi: 3.14159, steps: 150}
           expressions:
             z: "sin(kx*x) * sin(ky*y) * cos(w*t)"
           scalars:
             kx: 2.0
             ky: 3.0
             w: 1.0
           cmap: "RdBu_r"
           colorbar: true
           vmin: -1.0
           vmax: 1.0
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
    ParametricLinePlotGenerator : For 1D curves y(x, scalars)
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
        scalars: Mapping[str, Any] | None = None,
        # Convenience top-level scalar parameters (can be swept directly)
        t: float | None = None,
        kx: float | None = None,
        ky: float | None = None,
        w: float | None = None,
        x_label: str = "x",
        y_label: str = "y",
        title: str = "",
        figure_size: Tuple[float, float] = (8.0, 6.0),
        cmap: str = "viridis",
        colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        shading: Literal["auto", "flat", "nearest", "gouraud"] = "auto",
        aspect: float | Literal["auto", "equal"] = "equal",
    ) -> MatplotlibFigure:
        """
        Generate a 2D parametric surface plot from domain and expression.

        Parameters
        ----------
        domain : Mapping[str, Any]
            Domain specification with exactly two axes. Each axis defines
            'lo', 'hi', 'steps'.
            Example: ``{x: {lo: -1, hi: 1, steps: 100}, y: {lo: -1, hi: 1, steps: 100}}``
        expressions : Mapping[str, str]
            Mathematical expressions. Must contain key 'z' with expression
            in terms of domain axes and scalars.
            Example: ``{z: "sin(kx*x) * cos(ky*y)"}``
        scalars : Mapping[str, Any], optional
            Scalar parameters available in expressions.
            Example: ``{kx: 2.0, ky: 3.0, t: 0.0}``
        x_label : str, default "x"
            X-axis label
        y_label : str, default "y"
            Y-axis label
        title : str, default ""
            Plot title
        figure_size : Tuple[float, float], default (8.0, 6.0)
            Figure dimensions (width, height) in inches
        cmap : str, default "viridis"
            Matplotlib colormap name (e.g., "viridis", "plasma", "hot", "RdBu_r")
        colorbar : bool, default True
            Whether to display a colorbar
        vmin : float, optional
            Minimum value for colormap scaling. If None, uses data minimum.
        vmax : float, optional
            Maximum value for colormap scaling. If None, uses data maximum.
        shading : str, default "auto"
            Shading method for pcolormesh ("auto", "flat", "nearest", "gouraud")
        aspect : str, default "equal"
            Aspect ratio ("equal", "auto", or a number)

        Returns
        -------
        MatplotlibFigure
            The generated figure wrapped in MatplotlibFigure type

        Raises
        ------
        ValueError
            If domain doesn't have exactly 2 axes, expressions.z is missing,
            or expression evaluation fails

        Notes
        -----
        The figure is created without binding a GUI/backend-specific canvas.
        Rasterization processors attach an Agg canvas as needed for headless
        rendering. Use FigureToRGBAImage or FigureCollectionToRGBAStack with
        close_after=True to render and free resources, or rely on the
        automatic weakref finalizer in MatplotlibFigure.

        The surface is rendered using matplotlib's pcolormesh, which provides
        efficient rendering of 2D scalar fields.
        """
        # Create instance for base engine access
        instance = cls()

        # Parse and validate domain
        domain_specs = instance._base._parse_domain(
            domain
        )  # pylint: disable=protected-access
        if len(domain_specs) != 2:
            raise ValueError(
                f"ParametricSurfacePlotGenerator requires exactly 2 domain axes, "
                f"got {len(domain_specs)}: {list(domain_specs.keys())}"
            )

        # Build 2D grid
        axes_1d, axes_grid = instance._base._build_grid_2d(
            domain_specs
        )  # pylint: disable=protected-access
        axis_names = list(axes_1d.keys())

        # Validate expressions
        if "z" not in expressions:
            raise ValueError(
                "ParametricSurfacePlotGenerator requires expressions['z']. "
                f"Got expressions keys: {list(expressions.keys())}"
            )

        z_expr = expressions["z"]

        # Prepare scalars (empty dict if None), then overlay top-level params.
        # Top-level convenience parameters take precedence over entries in
        # the `scalars` mapping (consistent with ParametricLinePlotGenerator
        # and the documented behavior).
        scalar_params = dict(scalars) if scalars else {}
        if t is not None:
            scalar_params["t"] = t
        if kx is not None:
            scalar_params["kx"] = kx
        if ky is not None:
            scalar_params["ky"] = ky
        if w is not None:
            scalar_params["w"] = w

        # Evaluate z expression
        try:
            z_values = (
                instance._base._evaluate_expression(  # pylint: disable=protected-access
                    z_expr, axes_grid, scalar_params
                )
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to evaluate expressions['z'] = '{z_expr}': {e}"
            ) from e

        # Validate shape
        expected_shape = axes_grid[axis_names[0]].shape
        instance._base._validate_expression_shape(  # pylint: disable=protected-access
            z_values, expected_shape, "z", z_expr
        )

        # Create Figure instance directly (avoid pyplot state machine)
        fig = Figure(figsize=figure_size)
        ax = fig.add_subplot(111)

        # Plot surface using pcolormesh
        mesh = ax.pcolormesh(
            axes_grid[axis_names[0]],
            axes_grid[axis_names[1]],
            z_values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading=shading,
        )

        # Add colorbar if requested
        if colorbar:
            fig.colorbar(mesh, ax=ax, label="z")

        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # Set aspect ratio
        ax.set_aspect(aspect)

        # Wrap in MatplotlibFigure (lifecycle managed by weakref finalizer)
        return MatplotlibFigure(fig)
