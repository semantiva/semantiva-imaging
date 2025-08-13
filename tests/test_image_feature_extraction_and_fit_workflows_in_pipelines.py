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
from semantiva.workflows.fitting_model import PolynomialFittingModel
from semantiva.context_processors.context_processors import ModelFittingContextProcessor
from semantiva.pipeline import Pipeline
from semantiva.pipeline.payload import Payload
from semantiva_imaging.data_types import SingleChannelImageStack

from semantiva_imaging.probes import TwoDGaussianFitterProbe, SingleChannelImageProbe
from semantiva_imaging.data_io.loaders_savers import (
    TwoDGaussianSingleChannelImageGenerator,
    ParametricImageStackGenerator,
)
from semantiva.data_processors.data_slicer_factory import Slicer


class TwoDGaussianStdDevProbe(SingleChannelImageProbe):
    """A probe to extract the standard deviation of a 2D Gaussian from an image."""

    def _process_logic(self, data):
        gaussian_fitter_probe = TwoDGaussianFitterProbe()
        std_dev = gaussian_fitter_probe.process(data)["std_dev_x"]
        return std_dev


@pytest.fixture
def image_stack():
    """Fixture to provide a sample image stack with 3 frames using smaller images."""
    generator = ParametricImageStackGenerator(
        num_frames=3,
        parametric_expressions={
            "x_0": "50 + 5 * t",  # Adjusted peak x position to remain within a 128px image
            "y_0": "50 + 5 * t",  # Adjusted peak y position accordingly
            "std_dev": "10 + 2 * t",  # Reduced standard deviation for a smaller image
            "amplitude": "100",
        },
        param_ranges={"t": (-1, 2)},
        image_generator=TwoDGaussianSingleChannelImageGenerator(),
        image_generator_params={
            "image_size": (128, 128)
        },  # Use a much smaller image size
    )
    return generator.get_data(), generator.t_values


def test_pipeline_single_string_key(image_stack):
    """Test a pipeline with a single string key for dependent_var_key."""
    image_data, t_values = image_stack
    node_configurations = [
        {
            "processor": Slicer(TwoDGaussianStdDevProbe, SingleChannelImageStack),
            "context_keyword": "std_dev_features",
        },
        {
            "processor": ModelFittingContextProcessor,
            "parameters": {
                "fitting_model": PolynomialFittingModel(degree=1),
                "independent_var_key": "t_values",
                "dependent_var_key": "std_dev_features",
                "context_keyword": "std_dev_coefficients",
            },
        },
    ]
    pipeline = Pipeline(node_configurations)
    context_dict = {"t_values": t_values}
    payload_out = pipeline.process(Payload(image_data, context_dict))
    output_context = payload_out.context
    assert "std_dev_coefficients" in output_context.keys()


def test_pipeline_tuple_key(image_stack):
    """Test a pipeline with a tuple key for dependent_var_key."""
    image_data, t_values = image_stack
    node_configurations = [
        {
            "processor": Slicer(TwoDGaussianFitterProbe, SingleChannelImageStack),
            "context_keyword": "gaussian_fit_parameters",
        },
        {
            "processor": ModelFittingContextProcessor,
            "parameters": {
                "fitting_model": PolynomialFittingModel(degree=1),
                "independent_var_key": "t_values",
                "dependent_var_key": ("gaussian_fit_parameters", "std_dev_x"),
                "context_keyword": "std_dev_coefficients",
            },
        },
    ]
    pipeline = Pipeline(node_configurations)
    context_dict = {"t_values": t_values}
    payload_out = pipeline.process(Payload(image_data, context_dict))
    output_context = payload_out.context
    assert "std_dev_coefficients" in output_context.keys()
