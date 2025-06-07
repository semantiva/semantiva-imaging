import pytest
from semantiva.payload_operations import Pipeline
from semantiva.specializations import load_specializations
from semantiva_imaging.data_types import SingleChannelImage
from semantiva.configurations.load_pipeline_from_yaml import load_pipeline_from_yaml
from semantiva.context_processors.context_types import ContextType
from semantiva_imaging.data_io.loaders_savers import (
    ImageDataRandomGenerator,
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
    generator = ImageDataRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def context_type(random_image1):
    """Pytest fixture providing a dummy context."""
    return ContextType({"dummy": 1, "image_to_add": random_image1})


def test_pipeline_yaml(random_image1, yaml_config_path, context_type):
    """Test the pipeline processing using a YAML configuration file."""

    load_specializations("imaging")

    load_pipeline_config = load_pipeline_from_yaml(yaml_config_path)
    pipeline = Pipeline(load_pipeline_config)

    data, context = pipeline.process(random_image1, context_type)

    assert "final_info" in context.keys()
    assert "image_to_add" not in context.keys()
    assert isinstance(data, SingleChannelImage)
