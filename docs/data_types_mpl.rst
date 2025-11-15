.. _data_types_mpl:

Matplotlib Figure Types and Render Processors
=============================================

Overview
--------
Two new data types allow pipelines to carry matplotlib figures:

- ``MatplotlibFigure``: wraps a :class:`matplotlib.figure.Figure`
- ``MatplotlibFigureCollection``: a list-backed collection of ``MatplotlibFigure``

Render to Images
----------------
Use the processors below to rasterize figures to imaging-native types:

- ``FigureToRGBAImage`` → ``RGBAImage``
- ``FigureCollectionToRGBAStack`` → ``RGBAImageStack``

SingleChannelImage → MatplotlibFigure
--------------------------------------

In addition to consuming figures produced elsewhere, Semantiva imaging can
*create* Matplotlib figures from its own imaging data types.

Single-frame rendering
^^^^^^^^^^^^^^^^^^^^^^

Use :class:`semantiva_imaging.processing.SingleChannelImageToMatplotlibFigure`
to turn a :class:`semantiva_imaging.data_types.SingleChannelImage` into a
:class:`semantiva_imaging.data_types.mpl_figure.MatplotlibFigure`:

.. code-block:: yaml

   - processor: SingleChannelImageToMatplotlibFigure
     parameters:
       title: "Beam profile"
       colorbar: true
       cmap: "hot"
       log_scale: false
       xlabel: "x"
       ylabel: "y"

**Parameters mirror SingleChannelImageStackAnimator:**

- ``title`` (default ""): Figure title
- ``colorbar`` (default False): Add colorbar to figure
- ``cmap`` (default "hot"): Matplotlib colormap name
- ``log_scale`` (default False): Use logarithmic color scale
- ``xlabel`` (default ""): X-axis label
- ``ylabel`` (default ""): Y-axis label

Stack rendering (animations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To render an entire :class:`SingleChannelImageStack` (for example, from a
parametric sweep) into a :class:`MatplotlibFigureCollection` suitable for
animation:

.. code-block:: yaml

   - processor: SingleChannelImageStackToMatplotlibFigureCollection
     parameters:
       title: "Gaussian beam"
       colorbar: true
       cmap: "hot"
       log_scale: false
       xlabel: "x"
       ylabel: "y"

This processor applies the same visualization parameters to all frames in the
stack, creating a consistent animation sequence.

Complete Animation Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create an animated GIF from a SingleChannelImageStack, chain these processors:

.. code-block:: yaml

   pipeline:
     nodes:
       # 1. Generate or load SingleChannelImageStack
       - processor: TwoDGaussianSingleChannelImageGenerator
         derive:
           parameter_sweep:
             parameters:
               x_0: "50 + 55 * t"
               y_0: "150 + 5 * t + 5 * t ** 2"
             variables:
               t: { lo: -1, hi: 10, steps: 50 }
             collection: SingleChannelImageStack
         parameters:
           amplitude: 100

       # 2. Stack → MatplotlibFigureCollection
       - processor: SingleChannelImageStackToMatplotlibFigureCollection
         parameters:
           title: "Gaussian beam"
           colorbar: true
           cmap: "hot"

       # 3. FigureCollection → RGBAImageStack
       - processor: FigureCollectionToRGBAStack
         parameters:
           size_px: [1024, 1024]
           dpi: 100
           close_after: true

       # 4. Save as animated GIF
       - processor: AnimatedGifRGBAImageStackSaver
         parameters:
           path: "animation.gif"

See ``examples/gaussian_beam_figure_animation.yaml`` for a complete working example.

Parameters
^^^^^^^^^^
- ``size_px`` (required): ``[width_px, height_px]``
- ``dpi`` (default 100)
- ``transparent`` (default False)
- ``show_via_qt`` (default False) – best-effort; falls back headless-safe
- ``ion`` (default False) – interactive mode for live updates
- ``close_after`` (default False) – free memory after rendering

Best Practices
^^^^^^^^^^^^^^
- Author and style figures upstream. Rendering ops are for conversion.
- Prefer headless (Agg) in CI and batch; use Qt only when you need a window.
- Convert to ``RGBAImage`` before crossing process boundaries or persisting.

YAML Example
------------
.. code-block:: yaml

   nodes:
     - id: author_plot
       processor: MyDomainPlotter           # produces MatplotlibFigure
       output: MatplotlibFigure

     - id: fig_to_rgba
       processor: FigureToRGBAImage
       params:
         size_px: [1200, 800]
         dpi: 120
         transparent: false
         show_via_qt: false
         ion: false
         close_after: true
       input: author_plot
       output: RGBAImage

Python Example
--------------
.. code-block:: python

   import matplotlib.pyplot as plt
   from semantiva_imaging.data_types import MatplotlibFigure
   from semantiva_imaging.processing import FigureToRGBAImage

   # Create a simple plot
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.plot([0, 1, 2], [0, 1, 4])
   ax.set_title("Sample Plot")

   # Wrap in MatplotlibFigure
   mpl_fig = MatplotlibFigure(fig)

   # Rasterize to RGBA
   renderer = FigureToRGBAImage()
   rgba_image = renderer.process(
       mpl_fig,
       size_px=(1024, 768),
       dpi=120,
       transparent=False,
       close_after=True
   )

   # Now rgba_image is an RGBAImage that can be saved, processed, etc.
   print(rgba_image.data.shape)  # (768, 1024, 4)

Memory Management
-----------------
Matplotlib figures can consume significant memory. Consider:

- Use ``close_after=True`` to free figures after rendering
- Close figures manually with ``plt.close(fig)`` when done
- Avoid accumulating large collections of figures without rendering

Identity and Provenance
-----------------------
Matplotlib figures are considered ephemeral in Semantiva pipelines:

- Figures themselves are not serialized for cross-process execution
- Rendered ``RGBAImage`` outputs are the traceable, persistable artifacts
- All rendering parameters (size_px, dpi, etc.) are captured in execution traces

Headless Environments
---------------------
The rendering processors use the Agg backend by default, which works in headless
environments without display servers. The ``show_via_qt`` parameter is optional
and best-effort; if Qt is unavailable, rendering will still succeed headless.

Interactive Mode
----------------
Set ``ion=True`` to enable Matplotlib's interactive mode. This is useful for:

- Live updating of plots during development
- Debugging visualization issues
- Interactive exploration of data

Not recommended for production pipelines or CI environments.
