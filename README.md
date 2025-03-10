# Semantiva - Image specialization

## Overview

The **Semantiva Image Specialization** extends **Semantiva** with structured, type-safe image processing tools. It provides **image data types, processing algorithms, and visualization tools** designed for modular workflows.  

This specialization follows **Domain-Driven Design (DDD)** and **Type-Oriented Development**, ensuring consistency and flexibility for handling images and image stacks.

---

## Features

- **Structured Image Data Types**  
  - `ImageDataType`: Represents **2D images**  
  - `ImageStackDataType`: Represents **stacks of images** (3D)

- **Processing**  
  - **Arithmetic:** `ImageAddition`, `ImageSubtraction`
  - **Filtering & Normalization:** `ImageClipping`, `ImageNormalizerAlgorithm`
  - **Image Stack Projections:** `StackToImageMeanProjector`, `ImageStackToSideBySideProjector`

- **I/O and Image Generation**  
  - Load and save images in **PNG** and **NPZ** formats  
  - Generate synthetic images using `ImageDataRandomGenerator` and `TwoDGaussianImageGenerator`

- **Visualization (Jupyter Notebook Compatible)**  
  - **Interactive Cross-Section Viewer** (`ImageCrossSectionInteractiveViewer`) - Explore cross-sections of 2D images dynamically  
  - **Image Stack Animator** (`ImageStackAnimator`) - Animate sequences of stacked images  
  - **X-Y Projection Viewer** (`ImageXYProjectionViewer`) - View **intensity projections** along X and Y axes  
  - **Standard & Interactive Image Display** (`ImageViewer`, `ImageInteractiveViewer`)

- **Pipeline Support**  
  - Define and run **image processing workflows** using Semantiva’s pipeline system  

---

## Installation
```bash
pip install semantiva semantiva-imaging


## Overview


## Core Components

1. **Data Operations**
   - Abstract classes that enforce type-safe transformations, ensuring data flows remain coherent and domain-accurate.

2. **Context Operations**
   - Manages contextual or environmental information affecting data processing, enhancing adaptability and domain awareness.

3. **Payload Operations (Pipelines)**
   - Orchestrates the execution of multiple operations, combining data transformations and context adaptations into a coherent workflow.

4. **Data Types & Processing Types**
   - Defines the structure and constraints of domain-specific data, alongside compatible data processors (e.g., `Image` ↔ `ImageProcessor`), guaranteeing semantic integrity.

5. **Execution Tools**
   - Utilities for executing, monitoring, and debugging pipelines, supporting straightforward deployment and scaling.

## License

Semantiva is released under the [MIT License](./LICENSE), promoting collaborative development and broad adoption.

---

## Getting Started: A Parameterized Feature Extract-and-Fit Workflow

This is an advanced example demonstrating how Semantiva with Imaging specialization can generate images **based on metadata parameters**, extract features, and fit a simple model—all within a single pipeline. Notice how **context metadata** flows alongside **data**, allowing each operation to dynamically pull parameters from the context.

```python
from semantiva.logger import Logger
from semantiva_imaging.probes import (
    TwoDTiltedGaussianFitterProbe,
)
from semantiva.workflows.fitting_model import PolynomialFittingModel
from semantiva.context_processors.context_processors import ModelFittingContextProcessor
from semantiva.payload_operations.pipeline import Pipeline

from semantiva_imaging.data_io.loaders_savers import (
    TwoDGaussianImageGenerator,
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
    image_generator=TwoDGaussianImageGenerator(),
    image_generator_params={"image_size": (128, 128)},  # Image resolution
)

# Retrieve the generated stack of 2D images and the corresponding time values.
image_stack = generator.get_data()
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
        "processor": TwoDTiltedGaussianFitterProbe,
        # This probe extracts best-fit parameters for the 2D Gaussian in each frame
        # and stores them in the pipeline context under 'gaussian_fit_parameters'.
        "context_keyword": "gaussian_fit_parameters",
    },
    {
        "processor": ModelFittingContextProcessor,
        "parameters": {
            # Use a linear (degree=1) model to fit the extracted std_dev_x vs. t_values.
            "fitting_model": PolynomialFittingModel(degree=1),
            "independent_var_key": "t_values",
            "dependent_var_key": ("gaussian_fit_parameters", "std_dev_x"),
            "context_keyword": "std_dev_coefficients",
        },
    },
    {
        "processor": ModelFittingContextProcessor,
        "parameters": {
            # Also use a linear model to fit the orientation angle vs. t_values.
            "fitting_model": PolynomialFittingModel(degree=1),
            "independent_var_key": "t_values",
            "dependent_var_key": ("gaussian_fit_parameters", "angle"),
            "context_keyword": "orientation_coefficients",
        },
    },
]

# --- 3) Create and Run the Pipeline ---
pipeline = Pipeline(node_configurations)

# Pass the image stack (data) and the context dictionary (metadata) to the pipeline.
# Each pipeline step can read/write both data and context, enabling dynamic parameter injection.
output_data, output_context = pipeline.process(image_stack, context_dict)

# --- 4) Inspect Results ---
# 'std_dev_coefficients' and 'orientation_coefficients' were computed during pipeline execution.
# They store the best-fit linear coefficients for each feature.
print("Fitting Results for std_dev_x:",
      output_context.get_value("std_dev_coefficients"))
print("Fitting Results for orientation:",
      output_context.get_value("orientation_coefficients"))
```


---

## Acknowledgments

This framework draws inspiration from the rigorous demands of transparency and traceability in data-driven systems, particularly exemplified by the ALICE O2 project at CERN. The lessons learned from managing large-scale, high-throughput data in that environment—combined with the need for robust, domain-aligned workflows—shaped Semantiva’s emphasis on type-safe design, semantic clarity, and modular extensibility. By blending these concepts with principles of ontology-driven computing, Semantiva aims to deliver the same level of reliability and interpretability for any domain requiring advanced data processing and HPC integration.
