# Agent Guidelines for Semantiva Imaging

This repository houses **Semantiva Imaging**, the imaging extension of the Semantiva framework. It extends the core dual-channel pipeline with domain-specific data types, operations, I/O utilities and visualization helpers for working with images and image stacks.

## Repository Layout

* **semantiva_imaging/** - main package containing imaging utilities:
  * **data_types/** - `SingleChannelImage`, `SingleChannelImageStack`, `NChannelImage`, `RGBImage`, etc.
  * **data_io/** - loaders and savers (PNG/NPZ) and dummy generators.
  * **processing/** - image operations and pipeline processors.
  * **probes/** - Gaussian fitters and other image probes.
  * **visualization/** - viewers and stack animation utilities.
  * **README.md** - example usage of the imaging APIs.
* **examples/** - demonstration scripts such as the hyperspectral example.
* **tests/** - Pytest suite covering data types, I/O, operations, visualization, and pipelines.
* **docs/** - additional documentation assets used in the README.

The package is registered with Semantiva through `ImagingExtension`, enabling dynamic loading of its processors via the framework's `ClassRegistry`.

## Contribution Workflow

1. **Run static analysis and formatting**
Run the following commands from the repository root before committing:

```sh
black .
mypy semantiva-imaging/ tests/ examples/
```
pytest
```

2. **Documentation**
Update `README.md` or add docs under `docs/` when public APIs change or new imaging features are introduced. Record notable updates in `CHANGELOG.md` if present.