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

"""Tests for the n-channel processor factory functionality.

This module tests the _create_nchannel_processor factory function to ensure
that dynamically generated processor classes maintain proper functionality,
metadata exposure, and error handling behavior.
"""

import numpy as np
import inspect
import pytest

from semantiva_imaging.adapters.factory import _create_nchannel_processor
from semantiva_imaging.processing.base_nchannel import NChannelImageOperation
from semantiva_imaging.data_types import RGBImage, RGBAImage
from semantiva_imaging.data_types import NChannelImage


class _AddOp(NChannelImageOperation):
    """Add two n-channel images and scale the result.

    This is a minimal test operation used to verify the factory function's
    ability to wrap operations with multiple parameters, default values,
    and proper type handling.
    """

    def _process_logic(
        self, data: NChannelImage, other_image: NChannelImage, scale: float
    ):
        """Add two n-channel images and scale the result.

        Parameters
        ----------
        data : NChannelImage
            First input image (primary data)
        other_image : NChannelImage
            Second input image to add
        scale : float, optional
            Scaling factor applied to the sum,

        Returns
        -------
        NChannelImage
            Result of (data + other_image) * scale
        """
        return NChannelImage(
            (np.asarray(data.data) + np.asarray(other_image.data)) * scale,
            channel_info=data.channel_info,
        )


# Create a concrete processor for RGB images using the factory
# This demonstrates the intended usage pattern
AddRGBImageProcessor = _create_nchannel_processor(
    "AddRGBImageProcessor", _AddOp, RGBImage, RGBImage
)


def test_add_rgb_image_processor_functional():
    """Test that the generated processor correctly performs the underlying operation.

    This test verifies that:
    1. The generated processor can be instantiated
    2. It accepts the correct parameter types and counts
    3. It produces the expected mathematical result
    4. Default parameter values work correctly
    """
    # Create test data with known values for verification
    img1_arr = np.random.rand(2, 2, 3).astype(np.float32)
    img2_arr = np.random.rand(2, 2, 3).astype(np.float32) * 2
    img1 = RGBImage(img1_arr)
    img2 = RGBImage(img2_arr)

    # Instantiate the generated processor
    proc = AddRGBImageProcessor()

    # Test with explicit scale parameter
    out = proc.process(img1, img2, 2.0)

    # Verify the mathematical result matches expected computation
    assert np.all(out.data == (img1_arr + img2_arr) * 2.0)


def test_metadata_exposure():
    """Test that the generated processor exposes correct metadata for introspection.

    This test verifies that Semantiva's introspection system can properly
    extract parameter information, defaults, and type annotations from
    the generated processor class.
    """
    # Test auto-generated documentation
    assert AddRGBImageProcessor.__doc__.startswith(
        "Factory-adapted operator for _AddOp"
    )
    assert (
        "Add two n-channel images and scale the result" in AddRGBImageProcessor.__doc__
    )

    # Test that signature introspection works correctly
    sig = inspect.signature(AddRGBImageProcessor._process_logic)
    assert list(sig.parameters.keys()) == ["self", "data", "other_image", "scale"]

    # Test parameter name extraction for pipeline integration
    names = AddRGBImageProcessor.get_processing_parameter_names()
    assert names == ["other_image", "scale"]


def test_error_paths():
    """Test that the generated processor properly validates input types.

    This test ensures that:
    1. Type mismatches are caught and reported as TypeError
    2. Missing required parameters are handled correctly
    3. Error messages are informative
    """
    # Create test images of different types
    img1 = RGBImage(np.random.rand(2, 2, 3))  # RGB (3 channels)
    img2 = RGBAImage(np.random.rand(2, 2, 4))  # RGBA (4 channels)
    proc = AddRGBImageProcessor()

    # Test type mismatch: RGB processor should reject RGBA input
    with pytest.raises(TypeError):
        proc.process(img1, img2)

    # Test missing required parameter: should fail when other_image is not provided
    with pytest.raises(TypeError):
        proc.process(img1)
