import pytest
from semantiva.context_processors.context_types import (
    ContextType,
    ContextCollectionType,
)
from semantiva_imaging.data_io.loaders_savers import (
    ImageDataRandomGenerator,
    ImageStackRandomGenerator,
)
from semantiva_imaging.processing.operations import (
    ImageAddition,
    ImageSubtraction,
    ImageCropper,
    StackToImageMeanProjector,
)
from semantiva.payload_operations import Pipeline
from semantiva_imaging.data_types import (
    ImageDataType,
    ImageStackDataType,
)
from semantiva_imaging.probes import (
    BasicImageProbe,
)
from semantiva.data_processors.data_slicer_factory import Slicer


@pytest.fixture
def random_image():
    """
    Pytest fixture providing a random 2D ImageDataType instance.
    """
    generator = ImageDataRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def another_random_image():
    """
    Pytest fixture providing another random 2D ImageDataType instance.
    """
    generator = ImageDataRandomGenerator()
    return generator.get_data((256, 256))


@pytest.fixture
def random_image_stack():
    """
    Pytest fixture providing a random 3D ImageStackDataType instance (stack of 5 images).
    """
    generator = ImageStackRandomGenerator()
    return generator.get_data((5, 256, 256))  # Generates a stack of 5 images


@pytest.fixture
def random_context():
    """
    Pytest fixture providing a random ContextType instance.
    """
    return ContextType({"param": 42})


@pytest.fixture
def random_context_collection():
    """
    Pytest fixture providing a ContextCollectionType with 5 distinct context items.
    """
    return ContextCollectionType(
        context_list=[ContextType({"param": i}) for i in range(5)]
    )


def test_pipeline_slicing_with_single_context(
    random_image_stack, random_image, another_random_image, random_context
):
    """
    Tests slicing when using a single ContextType.

    - The `ImageStackDataType` is sliced into `ImageDataType` items.
    - The **same** ContextType instance is passed to each sliced item.
    - The final output should remain an `ImageStackDataType` with the same number of images.
    """

    node_configurations = [
        {
            "processor": Slicer(
                ImageAddition, ImageStackDataType
            ),  # Adds a specified image to each slice of the input data
            "parameters": {
                "image_to_add": random_image
            },  # Image to be added to each slice
        },
        {
            "processor": Slicer(
                ImageSubtraction, ImageStackDataType
            ),  # Subtracts a specified image from each slice of the input data
            "parameters": {
                "image_to_subtract": another_random_image
            },  # Image to subtract
        },
        {
            "processor": Slicer(
                BasicImageProbe, ImageStackDataType
            ),  # Probe operation to extract and store data
            "context_keyword": "mock_keyword",  # Stores probe results under this keyword in the context
            "parameters": {},  # No extra parameters required (can be omitted)
        },
        {
            "processor": Slicer(
                BasicImageProbe, ImageStackDataType
            ),  # Probe operation to collect results
            "parameters": {},  # No extra parameters required (can be omitted)
            # No `context_keyword`, making this node a ProbeCollectorNode (results stored internally)
        },
    ]

    pipeline = Pipeline(node_configurations)

    output_data, output_context = pipeline.process(random_image_stack, random_context)
    assert len(
        pipeline.get_probe_results()["Node 4/SlicerForBasicImageProbe"][0]
    ) == len(output_data)

    assert len(output_data) == len(
        output_context.get_value("mock_keyword")
    ), "Context for `mock_keyword` must contain one element per slice"

    assert isinstance(
        output_data, ImageStackDataType
    ), "Output should be an ImageStackDataType"
    assert len(output_data) == 5, "ImageStackDataType should retain 5 images"
    assert isinstance(
        output_context, ContextType
    ), "Context should remain a ContextType"


def test_pipeline_slicing_with_context_collection(
    random_image_stack, random_image, another_random_image, random_context_collection
):
    """
    Tests slicing when using a ContextCollectionType.

    - The `ImageStackDataType` is sliced into `ImageDataType` items.
    - A **corresponding** `ContextType` is used for each sliced item.
    - The final output should remain an `ImageStackDataType` with the same number of images.
    """

    node_configurations = [
        {
            "processor": Slicer(
                ImageAddition, ImageStackDataType
            ),  # Adds a specified image to each slice of the input data
            "parameters": {
                "image_to_add": random_image
            },  # Image to be added to each slice
        },
        {
            "processor": Slicer(
                ImageSubtraction, ImageStackDataType
            ),  # Subtracts a specified image from each slice of the input data
            "parameters": {
                "image_to_subtract": another_random_image
            },  # Image to subtract
        },
        {
            "processor": Slicer(
                BasicImageProbe, ImageStackDataType
            ),  # Probe operation to extract and store data
            "context_keyword": "mock_keyword",  # Stores probe results under this keyword in the context
            "parameters": {},  # No extra parameters required (can be omitted)
        },
        {
            "processor": Slicer(
                BasicImageProbe, ImageStackDataType
            ),  # Probe operation to collect results
            "parameters": {},  # No extra parameters required (can be omitted)
            # No `context_keyword`, making this node a ProbeCollectorNode (results stored internally)
        },
        {
            "processor": "rename:mock_keyword:renamed_keyword",  # Rename `mock_keyword` element to `renamed_keyword`
        },
        {
            "processor": "delete:renamed_keyword",  # Delete `renamed_keyword` from context
        },
    ]
    pipeline = Pipeline(node_configurations)

    output_data, output_context = pipeline.process(
        random_image_stack, random_context_collection
    )

    assert len(
        pipeline.get_probe_results()["Node 4/SlicerForBasicImageProbe"][0]
    ) == len(output_data)
    assert isinstance(
        output_data, ImageStackDataType
    ), "Output should be an ImageStackDataType"
    assert len(output_data) == 5, "ImageStackDataType should retain 5 images"
    assert isinstance(
        output_context, ContextCollectionType
    ), "Context should remain a ContextCollectionType"
    assert len(output_context) == 5, "ContextCollectionType should retain 5 items"


def test_pipeline_without_slicing(random_image, another_random_image, random_context):
    """
    Tests normal execution without slicing.

    - The pipeline receives a **single** `ImageDataType`, so no slicing occurs.
    - The entire image is processed in one pass.
    - The final output remains a **single** `ImageDataType`.
    """

    node_configurations = [
        {
            "processor": ImageAddition,
            "parameters": {"image_to_add": another_random_image},
        },
        {
            "processor": ImageCropper,
            "parameters": {"x_start": 50, "x_end": 200, "y_start": 50, "y_end": 200},
        },
    ]

    pipeline = Pipeline(node_configurations)

    output_data, output_context = pipeline.process(random_image, random_context)

    assert isinstance(output_data, ImageDataType), "Output should be an ImageDataType"
    assert isinstance(
        output_context, ContextType
    ), "Context should remain a ContextType"


def test_pipeline_invalid_slicing(random_image, random_context):
    """
    Tests invalid slicing scenario.

    - The pipeline expects an `ImageStackDataType` but receives `ImageDataType`.
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
        pipeline.process(random_image, random_context)
