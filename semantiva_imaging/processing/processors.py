from semantiva.data_processors import DataOperation, DataProbe
from ..data_types import SingleChannelImage, SingleChannelImageStack


class ImageOperation(DataOperation):
    """
    A DataOperation for :class:`SingleChannelImage` data.
    """

    @classmethod
    def input_data_type(cls):
        """
        Specify the input data type for the operation.

        Returns:
            type: `SingleChannelImage`, representing Image.
        """
        return SingleChannelImage

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImage]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `SingleChannelImage`, representing Image.
        """
        return SingleChannelImage


class SingleChannelImageStackAlgorithm(DataOperation):
    """
    A DataOperation for :class:`SingleChannelImageStack` data.
    """

    @classmethod
    def input_data_type(cls):
        """
        Specify the input data type for the operation.

        Returns:
            type: `SingleChannelImageStack`, representing a stack of images.
        """
        return SingleChannelImageStack

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImageStack]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `SingleChannelImageStack`, representing a stack of images.
        """
        return SingleChannelImageStack


class SingleChannelImageStackToImageProjector(DataOperation):
    """
    A DataOperation for flattening ``SingleChannelImageStack`` data into a ``SingleChannelImage``.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImageStack]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `SingleChannelImageStack`, representing a stack of images.
        """
        return SingleChannelImageStack

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImage]:
        """
        Specify the output data type for the operation.

        Returns:
            type: `SingleChannelImage`, representing Image.
        """
        return SingleChannelImage


class ImageProbe(DataProbe):
    """
    A DataProbe for :class:`SingleChannelImage` data.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImage]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `SingleChannelImage`, representing Image.
        """
        return SingleChannelImage


class SingleChannelImageStackProbe(DataProbe):
    """
    A DataProbe for :class:`SingleChannelImageStack` data.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImageStack]:
        """
        Specify the input data type for the operation.

        Returns:
            type: `SingleChannelImageStack`, representing a stack of images.
        """
        return SingleChannelImageStack
