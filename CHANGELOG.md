# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-11-16
### Added
- **Matplotlib figure types and rendering**
  - `MatplotlibFigure` and `MatplotlibFigureCollection` for lifecycle-safe figure management.
  - `FigureToRGBAImage` and `FigureCollectionToRGBAStack` for rendering figures to image and stack types.
- **Parametric plotting system**
  - `MatplotlibFigure`-based `ParametricLinePlotGenerator` (1D) and `ParametricSurfacePlotGenerator` (2D) using the domain/scalars/expressions model.
  - `BaseParametricPlotGenerator` and `ExpressionEnv` for shared domain parsing, grid generation, and safe expression evaluation.
- **Pipelines and examples**
  - Parameter sweep and run-space examples for parametric plotting.
  - `examples/wave_animation.yaml`: traveling wave animation using `sin(k*x - w*t)` and a time sweep.
  - `examples/parametric_surface_wave.yaml`: standing wave surface animation.
- **Documentation**
  - Guides for Matplotlib data types and parametric plots (`docs/data_types_mpl.rst`, `docs/parametric_plots.rst`).

## [0.2.0] - 2025-11-11
### Added
- **Core data types**
  - `SingleChannelImage` and `SingleChannelImageStack` with dtype validation and optional autocast.
  - `NChannelImage` / `NChannelImageStack`, plus concrete `RGBImage`, `RGBAImage` and their stack variants.
- **Processing operations**
  - Basic arithmetic and composition for single-channel images (e.g., addition, subtraction, cropping).
  - Stack projectors (e.g., mean) and side-by-side (horizontal) stack concatenation.
- **OpenCV-backed processors**
  - Factory-generated processors for common filters and transforms:
    - Filters: Gaussian, Median, Bilateral; edge detectors: Canny, Sobel, Laplacian.
    - Morphology: Dilate, Erode.
    - Transforms: Resize, Rotate, Flip.
    - Utilities: RGB→single-channel conversion, thresholding.
- **Data I/O**
  - Loaders & savers for `.npz`, PNG, JPEG, TIFF across single-channel and RGB/RGBA types.
  - Video support: write image stacks to AVI (multiple codecs attempted automatically).
  - GIF support: load/save animated GIFs for single-channel and RGB stacks.
  - URL loader utility to wrap existing loaders with HTTP/HTTPS download.
- **Visualization**
  - Matplotlib-based viewers for single images and stacks (colorbar, log-scale, colormap selection).
  - Interactive profile viewer (X/Y intensity profiles).
  - Simple stack animation utilities.
- **Probes**
  - Image probes for fitting 2D Gaussian models (including rotated/tilted variants) with R² diagnostics.

> Initial release of `semantiva-imaging`.
