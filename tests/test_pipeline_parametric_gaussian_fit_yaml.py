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

from semantiva.registry.plugin_registry import load_extensions
from semantiva.configurations.load_pipeline_from_yaml import load_pipeline_from_yaml
from semantiva.pipeline import Pipeline
from semantiva_imaging.data_types import SingleChannelImageStack


@pytest.fixture
def yaml_config_path():
    return "tests/pipeline_parametric_gaussian_fit.yaml"


def test_yaml_pipeline_parametric_gaussian_fit(yaml_config_path):
    load_extensions("semantiva-imaging")

    pipeline_config = load_pipeline_from_yaml(yaml_config_path)
    pipeline = Pipeline(pipeline_config)

    payload_out = pipeline.process()
    output_data, output_context = payload_out.data, payload_out.context

    # Validate outputs exist in context
    assert "gaussian_fit_parameters" in output_context.keys()
    assert "std_dev_coefficients" in output_context.keys()
    assert "orientation_coefficients" in output_context.keys()

    # Coefficients should be arrays or sequences of length degree+1 (here 2)
    std_coeffs = output_context.get_value("std_dev_coefficients")
    orient_coeffs = output_context.get_value("orientation_coefficients")

    assert hasattr(std_coeffs, "__len__") and len(std_coeffs) == 2
    assert hasattr(orient_coeffs, "__len__") and len(orient_coeffs) == 2

    # Ensure pipeline returns a stack (slicing should collect back to stack)
    assert isinstance(output_data, SingleChannelImageStack)

    # Validate fitted coefficients against expected parametric expressions
    # From YAML config: std_dev = "(50 + 20 * t, 20)" -> std_dev_x = 50 + 20*t
    # From YAML config: angle = "60 + 5 * t"

    # Check std_dev coefficients: should be 50 (intercept) + 20 (slope) * t
    assert (
        abs(std_coeffs["coeff_0"] - 50.0) < 0.000001
    ), f"std_dev coeff_0 {std_coeffs['coeff_0']} not close to 50.0"
    assert (
        abs(std_coeffs["coeff_1"] - 20.0) < 0.000001
    ), f"std_dev coeff_1 {std_coeffs['coeff_1']} not close to 20.0"

    # Check orientation coefficients: should be 60 (intercept) + 5 (slope) * t
    assert (
        abs(orient_coeffs["coeff_0"] - 60.0) < 0.000001
    ), f"orientation coeff_0 {orient_coeffs['coeff_0']} not close to 60.0"
    assert (
        abs(orient_coeffs["coeff_1"] - 5.0) < 0.000001
    ), f"orientation coeff_1 {orient_coeffs['coeff_1']} not close to 5.0"
