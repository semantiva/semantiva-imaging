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
from semantiva.pipeline import Pipeline
from semantiva.registry.plugin_registry import load_extensions
from semantiva_imaging.data_types import SingleChannelImage
from semantiva.configurations.load_pipeline_from_yaml import load_pipeline_from_yaml
from semantiva.context_processors.context_types import ContextType
from semantiva.pipeline.payload import Payload
from semantiva_imaging.data_io.loaders_savers import (
    SingleChannelImageRandomGenerator,
)


@pytest.fixture
def yaml_config_path():
    """Pytest fixture providing the path to a YAML configuration file."""
    return "tests/pipeline_config.yaml"


@pytest.fixture
def random_image1():
    """
    Pytest fixture for providing a random 2D SingleChannelImage instance using the dummy generator.
    """
    generator = SingleChannelImageRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def context_type(random_image1):
    """Pytest fixture providing a dummy context."""
    return ContextType({"dummy": 1, "image_to_add": random_image1})


def test_pipeline_yaml(random_image1, yaml_config_path, context_type):
    """Test the pipeline processing using a YAML configuration file."""

    load_extensions("semantiva_imaging")

    load_pipeline_config = load_pipeline_from_yaml(yaml_config_path)
    pipeline = Pipeline(load_pipeline_config)

    payload_out = pipeline.process(Payload(random_image1, context_type))
    data, context = payload_out.data, payload_out.context

    assert "final_info" in context.keys()
    assert "image_to_add" not in context.keys()
    assert isinstance(data, SingleChannelImage)
