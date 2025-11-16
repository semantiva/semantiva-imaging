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

import numpy as np
from semantiva_imaging.data_io.parametric_surface import ParametricSurfacePlotGenerator


def test_top_level_scalars_override_dict():
    """Verify top-level scalar parameters override entries in `scalars`.

    The generator should use the explicit top-level argument (t) instead of
    the value provided inside the `scalars` mapping.
    """
    fig_wrapper = ParametricSurfacePlotGenerator.get_data(
        domain={
            "x": {"lo": 0.0, "hi": 1.0, "steps": 10},
            "y": {"lo": 0.0, "hi": 1.0, "steps": 10},
        },
        # Use an expression that broadcasts to the 2D grid so the evaluated
        # z array has the expected (ny, nx) shape.
        expressions={"z": "t + 0*x"},
        scalars={"t": 1.5},
        t=3.25,
    )

    # Extract the Matplotlib Figure and the QuadMesh produced by pcolormesh
    fig = fig_wrapper.data
    assert fig.axes, "Figure must have at least one Axes"
    ax = fig.axes[0]
    assert ax.collections, "Axes must contain a QuadMesh collection"
    mesh = ax.collections[0]

    # get_array returns the flattened z-values used for colormapping
    values = mesh.get_array()
    mean_val = float(np.mean(values))

    assert np.allclose(mean_val, 3.25, rtol=1e-6, atol=1e-6)
