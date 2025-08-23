# Semantiva Imaging


## Overview
**Semantiva Imaging** is a demonstration extension of the Semantiva framework, showcasing sophisticated image processing capabilities and highlighting Semantiva’s domain-driven, type-centric philosophy. By maintaining a transparent workflow and flexible pipelines, it demonstrates how complex imaging tasks can be tackled with clarity, making Semantiva Imaging a powerful proof-of-concept for Semantiva’s innovative approach.

Visit the repositories:
- [Semantiva Imaging](https://github.com/semantiva/semantiva-imaging)
- [Semantiva Main](https://github.com/semantiva)

---

## Features

- **Structured Image Data Types**  
  - `SingleChannelImage`: Represents **2D images**
  - `SingleChannelImageStack`: Represents **stacks of images** (3D)

**Bit-depth handling** – `SingleChannelImage` and all colour types auto-cast 12-/16-bit arrays to `float32` by default. Pass `auto_cast=False` if you want to keep native 16-bit data and provide custom processors.

- **Processing**  
  - **Arithmetic:** `ImageAddition`, `ImageSubtraction`
  - **Filtering & Normalization:** `ImageCropper`, `ImageNormalizerOperation`
  - **Image Stack Projections:** `StackToImageMeanProjector`, `SingleChannelImageStackSideBySideProjector`
  - **Factory Utilities:** `_create_nchannel_processor` for wrapping raw N-channel algorithms

- **I/O and Image Generation**
  - Load and save images in **PNG**, **JPEG**, **TIFF**, and **NPZ** formats
  - Load images from HTTP(S) URLs via `UrlLoader`, e.g. `UrlLoader(PngRGBImageLoader)`
```python
from semantiva_imaging.data_io import PngRGBImageLoader
from semantiva_imaging.data_io.url_loader import UrlLoader

loader = UrlLoader(PngRGBImageLoader)
image = loader.get_data("https://avatars.githubusercontent.com/u/195345127?s=48") # Semantiva Github logo
```
  - Load and save **animated GIFs** with `AnimatedGifRGBAImageStackLoader` and `AnimatedGifSinglechannelImageStackSaver`
  - Generate synthetic images using `SingleChannelImageRandomGenerator` and `TwoDGaussianSingleChannelImageGenerator`
```
  - Save and load video stacks (`.avi`) and animated **GIFs**
  - Generate synthetic images using `SingleChannelImageRandomGenerator` and `TwoDGaussianSingleChannelImageGenerator`

- **Visualization (Jupyter Notebook Compatible)**  
  - **Interactive Cross-Section Viewer** (`ImageCrossSectionInteractiveViewer`) - Explore cross-sections of 2D images dynamically  
  - **Image Stack Animator** (`SingleChannelImageStackAnimator`) - Animate sequences of stacked images  
  - **X-Y Projection Viewer** (`ImageXYProjectionViewer`) - View **intensity projections** along X and Y axes  
  - **Standard & Interactive Image Display** (`ImageViewer`, `ImageInteractiveViewer`)

- **Pipeline Support**
  - Define and run **image processing workflows** using Semantiva’s pipeline system

### Color & Multichannel Image Types

`RGBImage` and `RGBAImage` now subclass the generic `NChannelImage`. They automatically
inherit the `auto_cast` behavior and carry `channel_info=("R","G","B")` or
`("R","G","B","A")`.

---

## Installation
```bash
pip install semantiva semantiva-imaging
```

## Getting Started: A Parameterized Feature Extract-and-Fit Workflow

This is an advanced example demonstrating how Semantiva with Imaging extension can generate images **based on metadata parameters**, extract features, and fit a simple model—all within a single pipeline. Notice how **context metadata** flows alongside **data**, allowing each operation to dynamically pull parameters from the context.

The parametrized generator bellow creates a stack of images with a time-varying 2D Gaussian signal. The signal's position, standard deviation, and orientation change over time. We then extract the Gaussian parameters from each frame and fit linear models to the standard deviation and orientation angle over time.


The following GIF shows the generated image stack, where the 2D Gaussian signal's position, standard deviation, and orientation change over time:

![](./docs/images/parametric_gaussian_signal.gif)



```python
from semantiva.logger import Logger
from semantiva_imaging.probes import (
    TwoDTiltedGaussianFitterProbe,
)
from semantiva_imaging.data_types.data_types import SingleChannelImageStack
from semantiva import ModelFittingContextProcessor, slicer
from semantiva.pipeline import Pipeline
from semantiva.pipeline.payload import Payload

from semantiva_imaging.data_io.loaders_savers import (
    TwoDGaussianSingleChannelImageGenerator,
    ParametricImageStackGenerator,
)

# --- 1) Parametric Image Generation ---
# We create a stack of images with a time-varying 2D Gaussian signal.
# 'ParametricImageStackGenerator' uses symbolic expressions to vary the Gaussian's position,
# standard deviation, angle, etc. over multiple frames (num_frames=10).
generator = ParametricImageStackGenerator(
    num_frames=3,
    parametric_expressions={
        "x_0": "50 + 5 * t",                # Time-dependent center x position
        "y_0": "50 + 5 * t + 5  * t ** 2",  # Time-dependent center y position
        "std_dev": "(50 + 20 * t, 20)",     # Stdev changes over frames
        "amplitude": "100",                 # Constant amplitude
        "angle": "60 + 5 * t",              # Orientation angle changes over frames
    },
    param_ranges={
        "t": (-1, 2)
    },  # 't' will sweep from -1 to +2, controlling the parametric expressions
    image_generator=TwoDGaussianSingleChannelImageGenerator(),
    image_generator_params={"image_size": (128, 128)},  # Image resolution
)

# Retrieve the generated stack of 2D images and the corresponding time values.
image_stack = generator.get_data() # See above the animation of this image stack

# Retrieve the 't' values used in generating the image stack.
t_values = generator.t_values  # List/array of 't' values used in generation.

# Prepare a context dictionary that includes 't_values' (the independent variable)
# for later use when fitting polynomial models to extracted features.
context_dict = {"t_values": t_values}

# --- 2) Define the Pipeline Configuration ---
# Our pipeline has three steps:
#   1. TwoDTiltedGaussianFitterProbe: Extracts Gaussian parameters (std_dev, angle, etc.) from each frame.
#   2. ModelFittingContextProcessor: Fits a polynomial model to the extracted std_dev_x feature vs. t_values.
#   3. Another ModelFittingContextProcessor: Fits a polynomial model to the extracted angle feature vs. t_values.
node_configurations = [
    {
        "processor": slicer(TwoDTiltedGaussianFitterProbe, SingleChannelImageStack),
        # This probe extracts best-fit parameters for the 2D Gaussian in each frame
        # and stores them in the pipeline context under 'gaussian_fit_parameters'.
        "context_keyword": "gaussian_fit_parameters",
    },
    {
        "processor": ModelFittingContextProcessor,
        "parameters": {
            # Use a linear (degree=1) model to fit the extracted std_dev_x vs. t_values.
            "fitting_model": "model:PolynomialFittingModel:degree=1",
            "independent_var_key": "t_values",
            "dependent_var_key": ("gaussian_fit_parameters", "std_dev_x"),
            "context_keyword": "std_dev_coefficients",
        },
    },
    {
        "processor": ModelFittingContextProcessor,
        "parameters": {
            # Also use a linear model to fit the orientation angle vs. t_values.
            "fitting_model": "model:PolynomialFittingModel:degree=1",
            "independent_var_key": "t_values",
            "dependent_var_key": ("gaussian_fit_parameters", "angle"),
            "context_keyword": "orientation_coefficients",
        },
    },
]

# --- 3) Create and Run the Pipeline ---
pipeline = Pipeline(node_configurations)

from semantiva.inspection import build_pipeline_inspection, summary_report
print("Pipeline inspection:")
inspection = build_pipeline_inspection(node_configurations)
print(summary_report(inspection))
for index, node in enumerate(pipeline.nodes, start=1):
    print(f"\nNode {index}")
    print(node.semantic_id())
# Pass the image stack (data) and the context dictionary (metadata) to the pipeline.
# Each pipeline step can read/write both data and context, enabling dynamic parameter injection.
payload_out = pipeline.process(Payload(image_stack, context_dict))
output_data, output_context = payload_out.data, payload_out.context

# --- 4) Inspect Results ---
# 'std_dev_coefficients' and 'orientation_coefficients' were computed during pipeline execution.
# They store the best-fit linear coefficients for each feature.
print("Fitting Results for std_dev_x:",
      output_context.get_value("std_dev_coefficients"))
print("Fitting Results for orientation:",
      output_context.get_value("orientation_coefficients"))
```

### Key Takeaways

* **Dual-Channel Processing**: Semantiva simultaneously processes **data** (the generated image stack) and **metadata** (like `t_values` and fitting parameters), ensuring each pipeline step can **dynamically** adapt based on evolving context.  
* **Parametric Generation & Feature Extraction**: You can generate synthetic images via symbolic expressions, then extract domain-specific features (e.g., Gaussian parameters) in one coherent workflow.  
* **Dynamic Parameter Injection**: Each node reads from and writes to a shared metadata context. That means you can modify or extend these parameters (e.g., changing the polynomial degree or image size) **without** altering code logic.  
* **Multi-Stage Modeling**: By chaining multiple `ModelFittingContextProcessor` steps, you can fit various features to different independent variables—particularly useful for research or production pipelines where multiple relationships must be modeled.  
* **Traceable & Auditable**: The final pipeline `context` retains the entire metadata history—including extracted features and fitted coefficients. This allows for transparent auditing, reproducibility, and potential handoff to subsequent pipelines or AI tools.

> With Semantiva’s **dual-channel** approach, you gain the flexibility to adapt pipeline logic on the fly. Even advanced tasks—such as parametric signal generation, feature extraction, and multi-stage model fitting—become modular, maintainable, and straightforward to extend.

### Codec-Dependent Classes

Some loader/saver classes in Semantiva Imaging depend on system-specific codecs, which may not be available or consistent across all environments. For detailed information about dependencies, risks, and recommendations, please refer to the [Codec Dependencies Documentation](./codec_dependencies.md).

## License

Semantiva-imaging is released under the [Apache License 2.0](./LICENSE).

