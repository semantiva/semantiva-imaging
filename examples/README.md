# Semantiva-Imaging Examples

This directory contains examples demonstrating **Semantiva's parameter sweep architectures** for building declarative, reproducible pipelines.

## Featured Examples

### 1. 📊 Collection-Based Sweep: `derive.parameter_sweep`

**File:**
- `param_sweep_pipeline.yaml` - Comprehensive, well-documented example

**What it demonstrates:**
- `derive.parameter_sweep` for node-level parameter sweep
- Automatic collection assembly (`MatplotlibFigureCollection`)
- Single pipeline execution → N items → downstream collection processing
- Best for: Batch operations, grouped analysis

**Key Concept:**
```
One Pipeline Execution
    ↓
ParametricPlotGenerator (sweep creates 5 figures)
    ↓
MatplotlibFigureCollection (grouped)
    ↓
FigureCollectionToRGBAStack (processes as one stack)
    ↓
Output: RGBAImageStack (5, 900, 1200, 4)
```

**Use Case:** You have N configurations and want to produce N items that you'll process together as a group (e.g., batch analysis, ensemble operations, combined visualization).

---

### 2. 🔀 Execution-Based Sweep: `run_space`

**File:**
- `run_space_plot_pipeline.yaml` - Full pipeline-level repetition example

**What it demonstrates:**
- `run_space` for pipeline-level parameter sweep
- Entire pipeline repeats with different context values (5 times)
- Each run: Single items through all nodes → independent output
- 5 independent PNG files, one per run
- Best for: Parameter studies, independent jobs, distributed execution

**Key Concept:**
```
Run-Space Context: 5 parameter sets
    ↓
Run 1: PlotGen → RenderFig → Path → SavePNG → plot_x_0(t).png
Run 2: PlotGen → RenderFig → Path → SavePNG → plot_y_0(t).png
Run 3: PlotGen → RenderFig → Path → SavePNG → plot_σ_x(t).png
Run 4: PlotGen → RenderFig → Path → SavePNG → plot_σ_y(t).png
Run 5: PlotGen → RenderFig → Path → SavePNG → plot_angle(t).png
```

**Use Case:** You have N configurations and want to run the entire pipeline independently N times with different parameters. Each run is self-contained and produces its own output.

---

## When to Use Each

| Need | Use | Pattern |
|------|-----|---------|
| N items grouped together | `derive.parameter_sweep` | Collection in single execution |
| N independent runs | `run_space` | Repeat pipeline N times |
| All results as one tensor | `derive.parameter_sweep` | FigureCollectionToRGBAStack |
| Each result separate file | `run_space` | One file per run |
| Process items as batch | `derive.parameter_sweep` | Batch statistics |
| Process each independently | `run_space` | Individual job |

---

## Key Differences

### Execution Model

**`derive.parameter_sweep`:**
- Processor executes **ONCE**
- Creates **collection** inside processor
- Subsequent nodes process **collection as single entity**
- Example: "Generate 5 figures, stack them, analyze stack"

**`run_space`:**
- Pipeline repeats **N times** (once per context)
- Each repetition processes **single items**
- Nodes see different context values per run
- Example: "Run pipeline 5 times with different configs, save each independently"

### Data Types

**`derive.parameter_sweep`:**
```
NoDataType → MatplotlibFigure (×5) → MatplotlibFigureCollection
  ↓
FigureCollectionToRGBAStack (expects collection)
  ↓
RGBAImageStack (single grouped output)
```

**`run_space`:**
```
Run 1: NoDataType → MatplotlibFigure → RGBAImage → (saved to file)
Run 2: NoDataType → MatplotlibFigure → RGBAImage → (saved to file)
...
Run 5: NoDataType → MatplotlibFigure → RGBAImage → (saved to file)
```

### Processor Requirements

**`derive.parameter_sweep`:**
- Output processor must handle **collection types**
- Example: `FigureCollectionToRGBAStack` (input: MatplotlibFigureCollection)

**`run_space`:**
- Processors handle **single types**
- Each node executes per-run with singles flowing through
- Example: `FigureToRGBAImage` (input: MatplotlibFigure, singular)

---

## Common Misconceptions

### ❌ "Both just do parameter sweeps, different syntax"

**Reality:** Different execution models.

- `derive`: Creates collection WITHIN a node (all items in memory together)
- `run_space`: Repeats pipeline ACROSS runs (each run independent)

Different data types flow through subsequent nodes.

### ❌ Using `run_space` expecting automatic collection

```yaml
# WRONG: run_space processes singles, not collections
run_space:
  context:
    equation: [5 equations]

pipeline:
  nodes:
    - processor: ParametricPlotGenerator
    - processor: FigureCollectionToRGBAStack  # ← Expects collection, gets single
```

**Fix:** Use `derive.parameter_sweep` with `collection` type if you want grouped output.

### ❌ Using `derive.parameter_sweep` for independent outputs

```yaml
# WRONG: derive produces one collection, not independent items
derive:
  parameter_sweep:
    collection: MatplotlibFigureCollection

pipeline:
  nodes:
    - processor: ParametricPlotGenerator
    - processor: PngImageSaver  # ← Tries to save one collection, not N items
```

**Fix:** Use `run_space` if you want each item saved independently.

---

## Architecture Principles

These examples demonstrate Semantiva's design:

### 1. Declarative Configuration
All pipeline logic in YAML. Processors are general-purpose, configured per use case.

### 2. Type-Driven Design
Processor output types constrain downstream processing.
- Collection output → must use collection-aware processors
- Single output → flows through standard processors

### 3. Multiple Execution Strategies
- `derive.parameter_sweep`: Within-node aggregation
- `run_space`: Pipeline-level repetition
- Combinations possible for complex workflows

### 4. Clear Separation of Concerns
- **Parameters**: Static values applied to all
- **Variables**: Swept values creating combinations
- **Context**: State shared across runs (in run_space)
- **Collection**: Grouped items (in derive.parameter_sweep)

---

## Running the Examples

Validate both examples work:

```bash
# Test derive.parameter_sweep (single execution → collection)
python -c "
from semantiva.pipeline import Pipeline
from semantiva.configurations.load_pipeline_from_yaml import load_pipeline_from_yaml
from semantiva.registry.plugin_registry import load_extensions

load_extensions('semantiva-imaging')
config = load_pipeline_from_yaml('examples/param_sweep_pipeline.yaml')
result = Pipeline(config).process()
print(f'Output type: {type(result.data).__name__}')
print(f'Stack shape: {result.data.data.shape}')  # Should be (5, 900, 1200, 4)
"

# Test run_space (N executions → independent outputs)
python -c "
from semantiva.pipeline import Pipeline
from semantiva.configurations.load_pipeline_from_yaml import load_pipeline_from_yaml
from semantiva.registry.plugin_registry import load_extensions
import os

load_extensions('semantiva-imaging')
config = load_pipeline_from_yaml('examples/run_space_plot_pipeline.yaml')
Pipeline(config).process()
count = len([f for f in os.listdir('run_space_plot_outputs') if f.endswith('.png')])
print(f'PNG files created: {count}')  # Should be 5
"
```

---

## Learn More

- **Run-Space Deep Dive**: See `RUNSPACE_MISCONCEPTION.md` for detailed explanation
- **Tests**: `../tests/test_parametric_plotter.py` for processor examples
- **Semantiva Core**: Main repository for framework architecture

---

## Summary

**Two complementary parameter sweep strategies:**

| Feature | `derive.parameter_sweep` | `run_space` |
|---------|------------------------|-----------|
| Execution count | 1 | N |
| Output structure | Collection | Singles (per-run) |
| Use for | Batch operations | Independent jobs |
| Output type | Grouped tensor | Separate files |
| Next processor | Collection-aware | Standard |

**Choose based on whether you want grouped or independent outputs.** ✨