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

"""Processors for rasterizing Matplotlib figures to RGBA images."""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from semantiva.data_processors import DataOperation
from semantiva_imaging.data_types.data_types import RGBAImage, RGBAImageStack
from semantiva_imaging.data_types.mpl_figure import (
    MatplotlibFigure,
    MatplotlibFigureCollection,
)


def _ensure_agg(fig: Figure) -> FigureCanvasAgg:
    """Bind an Agg canvas if none or non-Agg present."""
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    return fig.canvas  # type: ignore[return-value]


def _apply_size(fig: Figure, size_px: Tuple[int, int], dpi: int) -> None:
    """Apply size and DPI to figure."""
    w_px, h_px = size_px
    assert w_px > 0 and h_px > 0 and dpi > 0, "size_px and dpi must be positive"
    fig.set_dpi(dpi)
    fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)


def _apply_transparent(fig: Figure, transparent: bool) -> None:
    """Apply transparency to figure background."""
    if transparent:
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            if ax.patch is not None:
                ax.patch.set_alpha(0.0)


def _render_rgba(fig: Figure) -> np.ndarray:
    """Render figure to RGBA array."""
    canvas: FigureCanvasAgg = _ensure_agg(fig)
    canvas.draw()
    # buffer_rgba is supported by Agg; yields (H, W, 4) memoryview
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    return rgba


class FigureToRGBAImage(DataOperation):
    """Rasterize a MatplotlibFigure into an RGBAImage."""

    @classmethod
    def input_data_type(cls):
        return MatplotlibFigure

    @classmethod
    def output_data_type(cls):
        return RGBAImage

    def _process_logic(
        self,
        data: MatplotlibFigure,
        *,
        size_px: Tuple[int, int],
        dpi: int = 100,
        transparent: bool = False,
        close_after: bool = True,
    ) -> RGBAImage:
        """
        Rasterize a Matplotlib figure to an RGBA image using Agg backend.

        This processor renders figures in a non-interactive, headless-safe manner.
        Interactive visualization should be handled by separate viewer components
        or tools such as Semantiva Studio.

        Parameters
        ----------
        data : MatplotlibFigure
            The figure to rasterize.
        size_px : Tuple[int, int]
            Output size in pixels (width, height).
        dpi : int, default 100
            Dots per inch for rendering.
        transparent : bool, default False
            If True, make background transparent.
        close_after : bool, default True
            If True, close figure after rendering to free memory.
            Combined with the weakref finalizer in MatplotlibFigure,
            this ensures proper resource cleanup.

        Returns
        -------
        RGBAImage
            The rasterized RGBA image.
        """
        fig: Figure = data.data

        _apply_size(fig, size_px=size_px, dpi=dpi)
        _apply_transparent(fig, transparent=transparent)
        rgba = _render_rgba(fig)

        if close_after:
            plt.close(fig)

        return RGBAImage(rgba)


class FigureCollectionToRGBAStack(DataOperation):
    """Rasterize a collection of MatplotlibFigure into an RGBAImageStack."""

    @classmethod
    def input_data_type(cls):
        return MatplotlibFigureCollection

    @classmethod
    def output_data_type(cls):
        return RGBAImageStack

    def _process_logic(
        self,
        data: MatplotlibFigureCollection,
        *,
        size_px: Tuple[int, int],
        dpi: int = 100,
        transparent: bool = False,
        close_after: bool = True,
    ) -> RGBAImageStack:
        """
        Rasterize a collection of Matplotlib figures to an RGBA image stack.

        This processor renders figures in a non-interactive, headless-safe manner.
        All figures are rendered with uniform size and DPI settings.

        Parameters
        ----------
        data : MatplotlibFigureCollection
            The collection of figures to rasterize.
        size_px : Tuple[int, int]
            Output size in pixels (width, height) applied uniformly to all figures.
        dpi : int, default 100
            Dots per inch for rendering.
        transparent : bool, default False
            If True, make background transparent.
        close_after : bool, default True
            If True, close figures after rendering to free memory.

        Returns
        -------
        RGBAImageStack
            The rasterized RGBA image stack.
        """
        arrays: List[np.ndarray] = []
        for item in data:
            # Instantiate FigureToRGBAImage with the same context observer and logger
            renderer = FigureToRGBAImage(
                context_observer=self.context_observer, logger=self.logger
            )
            rgba = renderer._process_logic(
                item,
                size_px=size_px,
                dpi=dpi,
                transparent=transparent,
                close_after=close_after,
            ).data
            arrays.append(rgba)

        if len(arrays) == 0:
            # Return empty stack with correct shape
            h_px, w_px = size_px[1], size_px[0]
            stack = np.empty((0, h_px, w_px, 4), dtype=np.uint8)
        else:
            stack = np.stack(arrays, axis=0)
        return RGBAImageStack(stack)


# ============================================================================
# SingleChannelImage → MatplotlibFigure processors
# ============================================================================


def _render_single_channel_image_to_figure(
    image_data: np.ndarray,
    *,
    title: str = "",
    colorbar: bool = False,
    cmap: str = "hot",
    log_scale: bool = False,
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """
    Internal helper to render a 2D array as a Matplotlib Figure.

    This function mirrors the visualization behavior of SingleChannelImageStackAnimator
    to ensure consistent user experience between interactive and pipeline use.

    Parameters
    ----------
    image_data : np.ndarray
        2D array to visualize.
    title : str, default ""
        Figure title.
    colorbar : bool, default False
        If True, add a colorbar.
    cmap : str, default "hot"
        Matplotlib colormap name.
    log_scale : bool, default False
        If True, use logarithmic color scale.
    xlabel : str, default ""
        X-axis label.
    ylabel : str, default ""
        Y-axis label.

    Returns
    -------
    Figure
        Matplotlib Figure object with the rendered image.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    # Create figure without using pyplot state machine to avoid memory warnings
    fig = Figure()
    ax = fig.add_subplot(111)

    # Determine normalization
    norm = None
    if log_scale:
        vmin = image_data.min()
        vmax = image_data.max()
        # Avoid log(0) by setting a small positive floor
        norm = LogNorm(vmin=max(1e-34, vmin), vmax=vmax)

    # Display image
    im = ax.imshow(image_data, cmap=cmap, norm=norm, origin="lower")

    # Add colorbar if requested
    if colorbar:
        fig.colorbar(im, ax=ax)

    # Set labels and title
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig


class SingleChannelImageToMatplotlibFigure(DataOperation):
    """Render a SingleChannelImage as a MatplotlibFigure."""

    @classmethod
    def input_data_type(cls):
        from semantiva_imaging.data_types.data_types import SingleChannelImage

        return SingleChannelImage

    @classmethod
    def output_data_type(cls):
        return MatplotlibFigure

    def _process_logic(
        self,
        data,
        *,
        title: str = "",
        colorbar: bool = False,
        cmap: str = "hot",
        log_scale: bool = False,
        xlabel: str = "",
        ylabel: str = "",
    ) -> MatplotlibFigure:
        """
        Render a SingleChannelImage as a MatplotlibFigure.

        Parameters mirror the SingleChannelImageStackAnimator viewer to ensure
        consistent user experience between interactive and pipeline use.

        Parameters
        ----------
        data : SingleChannelImage
            The image to render.
        title : str, default ""
            Figure title.
        colorbar : bool, default False
            If True, add a colorbar.
        cmap : str, default "hot"
            Matplotlib colormap name (e.g., "hot", "viridis", "inferno").
        log_scale : bool, default False
            If True, use logarithmic color scale.
        xlabel : str, default ""
            X-axis label.
        ylabel : str, default ""
            Y-axis label.

        Returns
        -------
        MatplotlibFigure
            The rendered figure wrapped in MatplotlibFigure type.
        """
        fig = _render_single_channel_image_to_figure(
            data.data,
            title=title,
            colorbar=colorbar,
            cmap=cmap,
            log_scale=log_scale,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        return MatplotlibFigure(fig)


class SingleChannelImageStackToMatplotlibFigureCollection(DataOperation):
    """Render a SingleChannelImageStack as a MatplotlibFigureCollection."""

    @classmethod
    def input_data_type(cls):
        from semantiva_imaging.data_types.data_types import SingleChannelImageStack

        return SingleChannelImageStack

    @classmethod
    def output_data_type(cls):
        return MatplotlibFigureCollection

    def _process_logic(
        self,
        data,
        *,
        title: str = "",
        colorbar: bool = False,
        cmap: str = "hot",
        log_scale: bool = False,
        xlabel: str = "",
        ylabel: str = "",
    ) -> MatplotlibFigureCollection:
        """
        Render a SingleChannelImageStack as a MatplotlibFigureCollection.

        Applies the same visualization logic as SingleChannelImageToMatplotlibFigure
        to each frame in the stack, sharing parameters across all frames.

        Parameters
        ----------
        data : SingleChannelImageStack
            The image stack to render.
        title : str, default ""
            Figure title applied to all frames.
        colorbar : bool, default False
            If True, add a colorbar to each frame.
        cmap : str, default "hot"
            Matplotlib colormap name applied to all frames.
        log_scale : bool, default False
            If True, use logarithmic color scale for all frames.
        xlabel : str, default ""
            X-axis label applied to all frames.
        ylabel : str, default ""
            Y-axis label applied to all frames.

        Returns
        -------
        MatplotlibFigureCollection
            Collection of rendered figures, one per frame in the stack.
        """
        figures: List[MatplotlibFigure] = []

        # Iterate over frames in the stack
        for i in range(len(data)):
            # Get 2D slice for this frame
            frame_data = data.data[i]

            # Render this frame using the shared helper
            fig = _render_single_channel_image_to_figure(
                frame_data,
                title=title,
                colorbar=colorbar,
                cmap=cmap,
                log_scale=log_scale,
                xlabel=xlabel,
                ylabel=ylabel,
            )

            figures.append(MatplotlibFigure(fig))

        return MatplotlibFigureCollection(figures)
