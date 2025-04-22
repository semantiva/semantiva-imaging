from semantiva.data_processors import DataOperation, DataProbe
from ..data_types import ImageDataType, ImageStackDataType


class ImageOperation(DataOperation):
    """
    A DataOperation for ImageDataType data.
    """

    @classmethod
    def input_data_type(cls):
        """
        Specify the input data type for the operation.

        Returns:
            type: `ImageDataType`, representing Image.
        """
        return ImageDataType

    @classmethod
    def output_data_type(cls) -> type[ImageDataType]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `ImageDataType`, representing Image.
        """
        return ImageDataType


class ImageStackAlgorithm(DataOperation):
    """
    A DataOperation for ImageStackDataType data.
    """

    @classmethod
    def input_data_type(cls):
        """
        Specify the input data type for the operation.

        Returns:
            type: `ImageStackDataType`, representing a stack of images.
        """
        return ImageStackDataType

    @classmethod
    def output_data_type(cls) -> type[ImageStackDataType]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `ImageStackDataType`, representing a stack of images.
        """
        return ImageStackDataType


class ImageStackToImageProjector(DataOperation):
    """
    A DataOperation for flattening ImageStackDataType data into a ImageDataType.
    """

    @classmethod
    def input_data_type(cls) -> type[ImageStackDataType]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `ImageStackDataType`, representing a stack of images.
        """
        return ImageStackDataType

    @classmethod
    def output_data_type(cls) -> type[ImageDataType]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `ImageStackDataType`, representing a stack of images.
        """
        return ImageDataType


class ImageProbe(DataProbe):
    """
    A DataProbe for ImageDataType data.
    """

    @classmethod
    def input_data_type(cls) -> type[ImageDataType]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `ImageDataType`, representing Image.
        """
        return ImageDataType


class ImageStackProbe(DataProbe):
    """
    A DataProbe for ImageStackDataType data.
    """

    @classmethod
    def input_data_type(cls) -> type[ImageStackDataType]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `ImageStackDataType`, representing a stack of images.
        """
        return ImageStackDataType
