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
from semantiva.context_processors.context_types import (
    ContextType,
    ContextCollectionType,
)
from semantiva_imaging.data_io.loaders_savers import (
    SingleChannelImageRandomGenerator,
    SingleChannelImageStackRandomGenerator,
)
from semantiva_imaging.processing.operations import (
    ImageAddition,
    ImageCropper,
    StackToImageMeanProjector,
)
from semantiva.pipeline import Pipeline
from semantiva.pipeline.payload import Payload
from semantiva_imaging.data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
)


@pytest.fixture
def random_image():
    """
    Pytest fixture providing a random 2D SingleChannelImage instance.
    """
    generator = SingleChannelImageRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def another_random_image():
    """
    Pytest fixture providing another random 2D SingleChannelImage instance.
    """
    generator = SingleChannelImageRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def random_image_stack():
    """
    Pytest fixture providing a random 3D SingleChannelImageStack instance (stack of 5 images).
    """
    generator = SingleChannelImageStackRandomGenerator()
    return generator.get_data((5, 256, 256))  # Generates a stack of 5 images


@pytest.fixture
def random_context(random_image, another_random_image):
    """
    Pytest fixture providing a random ContextType instance.
    """
    return ContextType(
        {
            "param": 42,
            "image_to_add": random_image,
            "image_to_subtract": another_random_image,
        }
    )


@pytest.fixture
def random_context_collection(random_image, another_random_image):
    """
    Pytest fixture providing a ContextCollectionType with 5 distinct context items.
    """
    return ContextCollectionType(
        context_list=[ContextType({"param": i}) for i in range(5)],
        global_context={
            "image_to_add": random_image,
            "image_to_subtract": another_random_image,
        },
    )


def test_pipeline_slicing_with_single_context(
    random_image_stack, random_image, another_random_image, random_context
):
    """
    Tests slicing when using a single ContextType.

    - The `SingleChannelImageStack` is sliced into `SingleChannelImage` items.
    - The **same** ContextType instance is passed to each sliced item.
    - The final output should remain an `SingleChannelImageStack` with the same number of images.
    """

    node_configurations = [
        {
            "processor": "slice:ImageAddition:SingleChannelImageStack",
            # Adds a specified image to each slice of the input data
        },
        {
            "processor": "slice:ImageSubtraction:SingleChannelImageStack",
            # Subtracts a specified image from each slice of the input data
        },
        {
            "processor": "slice:BasicImageProbe:SingleChannelImageStack",
            # Probe operation to extract and store data
            "context_key": "mock_keyword",  # Stores probe results under this key in the context
            "parameters": {},  # No extra parameters required (can be omitted)
        },
        {
            "processor": "slice:BasicImageProbe:SingleChannelImageStack",
            # Probe operation to collect results
            "parameters": {},  # No extra parameters required (can be omitted)
            "context_key": "mock_key",
        },
    ]

    pipeline = Pipeline(node_configurations)

    payload_out = pipeline.process(Payload(random_image_stack, random_context))
    output_data, output_context = payload_out.data, payload_out.context

    assert len(output_data) == len(
        output_context.get_value("mock_keyword")
    ), "Context for `mock_keyword` must contain one element per slice"

    assert isinstance(
        output_data, SingleChannelImageStack
    ), "Output should be an SingleChannelImageStack"
    assert len(output_data) == 5, "SingleChannelImageStack should retain 5 images"
    assert isinstance(
        output_context, ContextType
    ), "Context should remain a ContextType"


def test_pipeline_slicing_with_context_collection(
    random_image_stack, random_image, another_random_image, random_context_collection
):
    """
    Tests slicing when using a ContextCollectionType.

    - The `SingleChannelImageStack` is sliced into `SingleChannelImage` items.
    - A **corresponding** `ContextType` is used for each sliced item.
    - The final output should remain an `SingleChannelImageStack` with the same number of images.
    """

    node_configurations = [
        {
            "processor": "slice:ImageAddition:SingleChannelImageStack",  # Adds a specified image to each slice of the input data
        },
        {
            "processor": "slice:ImageSubtraction:SingleChannelImageStack",  # Subtracts a specified image from each slice of the input data
        },
        {
            "processor": "slice:BasicImageProbe:SingleChannelImageStack",  # Probe operation to extract and store data
            "context_key": "mock_keyword",  # Stores probe results under this key in the context
        },
        {
            "processor": "slice:BasicImageProbe:SingleChannelImageStack",  # Probe operation to collect results
            "context_key": "mock_key",
        },
        {
            "processor": "rename:mock_keyword:renamed_keyword",  # Rename `mock_keyword` element to `renamed_keyword`
        },
        {
            "processor": "delete:renamed_keyword",  # Delete `renamed_keyword` from context
        },
    ]
    pipeline = Pipeline(node_configurations)

    payload_out = pipeline.process(
        Payload(random_image_stack, random_context_collection)
    )
    output_data, output_context = payload_out.data, payload_out.context

    assert isinstance(
        output_data, SingleChannelImageStack
    ), "Output should be an SingleChannelImageStack"
    assert len(output_data) == 5, "SingleChannelImageStack should retain 5 images"
    assert isinstance(
        output_context, ContextCollectionType
    ), "Context should remain a ContextCollectionType"
    assert len(output_context) == 5, "ContextCollectionType should retain 5 items"


def test_pipeline_without_slicing(random_image, random_context):
    """
    Tests normal execution without slicing.

    - The pipeline receives a **single** `SingleChannelImage`, so no slicing occurs.
    - The entire image is processed in one pass.
    - The final output remains a **single** `SingleChannelImage`.
    """

    node_configurations = [
        {
            "processor": ImageAddition,
        },
        {
            "processor": ImageCropper,
            "parameters": {"x_start": 50, "x_end": 200, "y_start": 50, "y_end": 200},
        },
    ]

    pipeline = Pipeline(node_configurations)

    payload_out = pipeline.process(Payload(random_image, random_context))
    output_data, output_context = payload_out.data, payload_out.context

    assert isinstance(
        output_data, SingleChannelImage
    ), "Output should be an SingleChannelImage"
    assert isinstance(
        output_context, ContextType
    ), "Context should remain a ContextType"


def test_pipeline_invalid_slicing(random_image, random_context):
    """
    Tests invalid slicing scenario.

    - The pipeline expects an `SingleChannelImageStack` but receives `SingleChannelImage`.
    - This should raise a **TypeError** due to incompatible pipeline topology.
    """

    node_configurations = [
        {
            "processor": StackToImageMeanProjector,
            "parameters": {},
        },
    ]

    pipeline = Pipeline(node_configurations)

    with pytest.raises(TypeError):
        pipeline.process(Payload(random_image, random_context))
