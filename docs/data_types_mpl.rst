.. _data_types_mpl:

Matplotlib Figure Types and Rendering Principles
================================================

Overview
--------
This document describes how Semantiva Imaging models, renders, and converts
Matplotlib figures. The core building blocks are:

- ``MatplotlibFigure``: wraps a :class:`matplotlib.figure.Figure` and adds
  automatic lifecycle safety (see :ref:`figure-lifecycle`).
- ``MatplotlibFigureCollection``: a typed collection for aggregating
  ``MatplotlibFigure`` instances (e.g., for animation pipelines).

Lifecycle Safety
----------------

``MatplotlibFigure`` registers a weakref finalizer that closes the underlying
Matplotlib figure when the wrapper is garbage-collected. Rendering processors
also expose ``close_after=True`` so you can deterministically release
resources right after rasterization. Together, these mechanisms keep
pipelines from leaking Matplotlib figures even in long-lived or high-throughput
executions.

Rendering Processors (Figure → Image)
-------------------------------------

Rasterization is handled by two processors:

- ``FigureToRGBAImage`` → ``RGBAImage``
- ``FigureCollectionToRGBAStack`` → ``RGBAImageStack``

Both render via the Agg backend (headless-safe, non-interactive), support
``size_px``, ``dpi``, and ``transparent`` parameters, and default
``close_after=True`` to free the figures once drawing completes. Interactive GUI
hooks (Qt, ``plt.ion``) are intentionally omitted; visualization belongs to
dedicated viewers like ``SingleChannelImageStackAnimator``.

Rendering Parameters Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``size_px`` (required): ``[width_px, height_px]``
- ``dpi`` (default 100): Rendering resolution
- ``transparent`` (default False): Transparent background
- ``close_after`` (default True): Free memory immediately after rendering
