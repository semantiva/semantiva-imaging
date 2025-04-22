from abc import abstractmethod
from typing_extensions import override
from semantiva.context_processors.context_types import ContextType
from semantiva.data_io import DataSource, PayloadSource, DataSink, PayloadSink
from ..data_types import ImageDataType, ImageStackDataType


class ImageDataSource(DataSource):
    """
    Abstract base class for image data sources.
    """

    @abstractmethod
    def _get_data(self, *args, **kwargs) -> ImageDataType:
        """
        Abstract method to retrieve `ImageDataType` data from the source.

        Subclasses must implement this method to provide a specific mechanism for retrieving
        image data.

        Returns:
            ImageDataType: The retrieved image data.
        """
        pass

    @classmethod
    def get_data(cls, *args, **kwargs) -> ImageDataType:
        """
        Fetch and return `ImageDataType` data.

        This method calls the subclass-implemented `_get_data` method to retrieve the data.

        Returns:
            ImageDataType: The fetched image data.
        """
        return cls()._get_data(*args, **kwargs)

    @classmethod
    def output_data_type(cls) -> type[ImageDataType]:
        return ImageDataType


class ImageStackSource(DataSource):
    """
    Abstract base class for image stack data sources.
    """

    @abstractmethod
    def _get_data(self, *args, **kwargs) -> ImageStackDataType:
        """
        Abstract method to retrieve `ImageStackDataType` data from the source.

        Subclasses must implement this method to provide a specific mechanism for retrieving
        image stack data.

        Returns:
            ImageStackDataType: The retrieved image stack data.
        """

    def get_data(self, *args, **kwargs) -> ImageStackDataType:
        """
        Fetch and return `ImageStackDataType` data.

        This method calls the subclass-implemented `_get_data` method to retrieve the data.

        Returns:
            ImageStackDataType: The fetched image stack data.
        """
        return self._get_data(*args, **kwargs)

    @classmethod
    def output_data_type(cls) -> type[ImageStackDataType]:  # type: ignore
        return ImageStackDataType


class ImageDataSink(DataSink):
    """
    Abstract base class for image data sinks.
    """

    @abstractmethod
    def _send_data(self, data: ImageDataType, *args, **kwargs):
        """
        Abstract method to consume and store `ImageDataType` data.

        Subclasses must implement this method to define the mechanism for consuming and
        storing image data.

        Parameters:
            data (ImageDataType): The image data to be consumed or stored.
        """

    def send_data(self, data: ImageDataType, *args, **kwargs):
        """
        Consume and store `ImageDataType` data.

        This method calls the subclass-implemented `_send_data` method to process the data.

        Parameters:
            data (ImageDataType): The image data to be consumed or stored.
        """
        self._send_data(data, *args, **kwargs)

    def input_data_type(self) -> type[ImageDataType]:  # type: ignore
        return ImageDataType


class ImageStackDataSink(DataSink):
    """
    Abstract base class for image stack data sinks.
    """

    @abstractmethod
    def _send_data(self, data: ImageStackDataType, *args, **kwargs):
        """
        Abstract method to consume and store `ImageStackDataType` data.

        Subclasses must implement this method to define the mechanism for consuming and
        storing image stack data.

        Parameters:
            data (ImageStackDataType): The image stack data to be consumed or stored.
        """

    def send_data(self, data: ImageStackDataType, *args, **kwargs):
        """
        Consume and store `ImageStackDataType` data.

        This method calls the subclass-implemented `_send_data` method to process the data.

        Parameters:
            data (ImageStackDataType): The image stack data to be consumed or stored.
        """
        self._send_data(data, *args, **kwargs)

    def input_data_type(self) -> type[ImageStackDataType]:  # type: ignore
        return ImageStackDataType


class ImagePayloadSink(PayloadSink):
    """
    Abstract base class for sinks that consume and store `ImageDataType` objects with associated context.
    """

    @abstractmethod
    @override
    def _send_payload(self, data: ImageDataType, context: ContextType, *args, **kwargs):
        """
        Abstract method to consume and store `ImageDataType` data along with context.

        Subclasses must implement this method to define the mechanism for consuming and
        storing the data and context.

        Parameters:
            data (ImageDataType): The image data to be consumed or stored.
            context (ContextType): The associated context or metadata for the image data.
        """

    def send_payload(self, data, context, *args, **kwargs):
        """
        Consume and store `ImageDataType` data along with context.

        This method calls the subclass-implemented `_send_payload` method to process the data and context.

        Parameters:
            data (ImageDataType): The image data to be consumed or stored.
            context (dict): The associated context or metadata for the image data.
        """
        self._send_payload(data, context, *args, **kwargs)

    def input_data_type(self) -> type[ImageDataType]:  # type: ignore
        """
        Returns the expected input data type for the data.

        Returns:
            type: `ImageDataType`, the expected type for the data parameter.
        """
        return ImageDataType


class ImageStackPayloadSource(PayloadSource):
    """
    Abstract base class for sources that provide `ImageStackDataType` objects with associated context.
    """

    @abstractmethod
    def _get_payload(self, *args, **kwargs) -> tuple[ImageStackDataType, ContextType]:
        """
        Abstract method to retrieve an `ImageStackDataType` object and its associated context.

        Subclasses must implement this method to define the mechanism for retrieving the data and context.

        Returns:
            tuple[ImageStackDataType, dict]:
                A tuple where the first element is the `ImageStackDataType` object and
                the second element is a dictionary representing the context or metadata.
        """

    def get_payload(self, *args, **kwargs) -> tuple[ImageStackDataType, ContextType]:
        """
        Fetch and return an `ImageStackDataType` object and its associated context.

        This method calls the subclass-implemented `_get_payload` method to retrieve the data and context.

        Returns:
            tuple[ImageStackDataType, dict]:
                A tuple where the first element is the `ImageStackDataType` object and
                the second element is a dictionary representing the context or metadata.
        """
        return self._get_payload(*args, **kwargs)

    def output_data_type(self) -> type[ImageStackDataType]:  # type: ignore
        """
        Returns the expected output data type for the data.

        Returns:
            type: `ImageStackDataType`, the expected type for the data part of the payload.
        """
        return ImageStackDataType
