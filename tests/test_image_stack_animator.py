import pytest
import numpy as np
from semantiva_imaging.data_types import SingleChannelImageStack
from semantiva_imaging.visualization.viewers import SingleChannelImageStackAnimator


# Sample test data
@pytest.fixture
def test_image_stack():
    """Fixture to provide test image stack data."""
    return SingleChannelImageStack(np.random.rand(10, 256, 256))


def test_display_animation(test_image_stack):
    """Test that the animation is correctly displayed."""
    SingleChannelImageStackAnimator.view(test_image_stack)
