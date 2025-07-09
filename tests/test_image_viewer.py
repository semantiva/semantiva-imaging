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

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from semantiva_imaging.visualization.viewers import ImageViewer
from semantiva_imaging.data_types import SingleChannelImage


@pytest.fixture
def test_image():
    """Fixture to provide test image data."""
    return SingleChannelImage(data=np.random.rand(10, 10))


def test_generate_image(test_image):
    """Test if _generate_image() creates a valid figure."""
    fig = ImageViewer._generate_image(
        test_image,
        title="Test",
        colorbar=True,
        cmap="plasma",
        log_scale=True,
        xlabel="X",
        ylabel="Y",
    )

    assert isinstance(fig, plt.Figure)  # Ensure a matplotlib Figure is returned
    assert len(fig.axes) > 0  # Ensure the figure has at least one axis

    ax = fig.axes[0]  # Get the first axis
    assert ax.get_title() == "Test"  # Check title
    assert ax.get_xlabel() == "X"  # Check x-axis label
    assert ax.get_ylabel() == "Y"  # Check y-axis label

    # Check colormap
    assert ax.images[0].get_cmap().name == "plasma"

    # Check if log scale is applied
    assert isinstance(ax.images[0].norm, LogNorm)
