from abc import abstractmethod
from typing_extensions import override
from semantiva.context_processors.context_types import ContextType
from semantiva.data_io import DataSource, PayloadSource, DataSink, PayloadSink
from ..data_types import SingleChannelImage, SingleChannelImageStack


class SingleChannelImageDataSource(DataSource):
    """
    Abstract base class for image data sources.
    """

    @classmethod
    @abstractmethod
    def _get_data(cls, *args, **kwargs):
        """
        Abstract method to retrieve `SingleChannelImage` data from the source.

        Subclasses must implement this method to provide a specific mechanism for retrieving
        image data.

        Returns:
            SingleChannelImage: The retrieved image data.
        """
        pass

    @classmethod
    def get_data(cls, *args, **kwargs):
        """
        Fetch and return `SingleChannelImage` data.

        This method calls the subclass-implemented `_get_data` method to retrieve the data.

        Returns:
            SingleChannelImage: The fetched image data.
        """
        return cls()._get_data(*args, **kwargs)

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImage]:
        return SingleChannelImage


class SingleChannelImageStackSource(DataSource):
    """
    Abstract base class for image stack data sources.
    """

    @classmethod
    @abstractmethod
    def _get_data(cls, *args, **kwargs):
        """
        Abstract method to retrieve `SingleChannelImageStack` data from the source.

        Subclasses must implement this method to provide a specific mechanism for retrieving
        image stack data.

        Returns:
            SingleChannelImageStack: The retrieved image stack data.
        """

    def get_data(self, *args, **kwargs):
        """
        Fetch and return `SingleChannelImageStack` data.

        This method calls the subclass-implemented `_get_data` method to retrieve the data.

        Returns:
            SingleChannelImageStack: The fetched image stack data.
        """
        return self._get_data(*args, **kwargs)

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImageStack]:  # type: ignore
        return SingleChannelImageStack


class SingleChannelImageDataSink(DataSink):
    """
    Abstract base class for image data sinks.
    """

    @abstractmethod
    def _send_data(self, data: SingleChannelImage, *args, **kwargs):
        """
        Abstract method to consume and store `SingleChannelImage` data.

        Subclasses must implement this method to define the mechanism for consuming and
        storing image data.

        Parameters:
            data (SingleChannelImage): The image data to be consumed or stored.
        """

    def send_data(self, data: SingleChannelImage, *args, **kwargs):
        """
        Consume and store `SingleChannelImage` data.

        This method calls the subclass-implemented `_send_data` method to process the data.

        Parameters:
            data (SingleChannelImage): The image data to be consumed or stored.
        """
        self._send_data(data, *args, **kwargs)

    def input_data_type(self) -> type[SingleChannelImage]:  # type: ignore
        return SingleChannelImage


class SingleChannelImageStackSink(DataSink):
    """
    Abstract base class for SingleChannelImageStack sinks.
    """

    @abstractmethod
    def _send_data(self, data: SingleChannelImageStack, *args, **kwargs):
        """
        Abstract method to consume and store `SingleChannelImageStack` data.

        Subclasses must implement this method to define the mechanism for consuming and
        storing SingleChannelImageStack data.

        Parameters:
            data (SingleChannelImageStack): The image stack data to be consumed or stored.
        """

    def send_data(self, data: SingleChannelImageStack, *args, **kwargs):
        """
        Consume and store `SingleChannelImageStack` data.

        This method calls the subclass-implemented `_send_data` method to process the data.

        Parameters:
            data (SingleChannelImageStack): The image stack data to be consumed or stored.
        """
        self._send_data(data, *args, **kwargs)

    def input_data_type(self) -> type[SingleChannelImageStack]:  # type: ignore
        return SingleChannelImageStack


class ImagePayloadSink(PayloadSink):
    """
    Abstract base class for sinks that consume and store ``SingleChannelImage`` objects with associated context.
    """

    @abstractmethod
    @override
    def _send_payload(
        self, data: SingleChannelImage, context: ContextType, *args, **kwargs
    ):
        """
        Abstract method to consume and store `SingleChannelImage` data along with context.

        Subclasses must implement this method to define the mechanism for consuming and
        storing the data and context.

        Parameters:
            data (SingleChannelImage): The image data to be consumed or stored.
            context (ContextType): The associated context or metadata for the image data.
        """

    def send_payload(self, data, context, *args, **kwargs):
        """
        Consume and store `SingleChannelImage` data along with context.

        This method calls the subclass-implemented `_send_payload` method to process the data and context.

        Parameters:
            data (SingleChannelImage): The image data to be consumed or stored.
            context (dict): The associated context or metadata for the image data.
        """
        self._send_payload(data, context, *args, **kwargs)

    def input_data_type(self) -> type[SingleChannelImage]:  # type: ignore
        """
        Returns the expected input data type for the data.

        Returns:
            type: `SingleChannelImage`, the expected type for the data parameter.
        """
        return SingleChannelImage


class SingleChannelImageStackPayloadSource(PayloadSource):
    """
    Abstract base class for sources that provide ``SingleChannelImageStack`` objects with associated context.
    """

    @abstractmethod
    def _get_payload(
        self, *args, **kwargs
    ) -> tuple[SingleChannelImageStack, ContextType]:
        """
        Abstract method to retrieve an `SingleChannelImageStack` object and its associated context.

        Subclasses must implement this method to define the mechanism for retrieving the data and context.

        Returns:
            tuple[SingleChannelImageStack, dict]:
                A tuple where the first element is the `SingleChannelImageStack` object and
                the second element is a dictionary representing the context or metadata.
        """

    def get_payload(
        self, *args, **kwargs
    ) -> tuple[SingleChannelImageStack, ContextType]:
        """
        Fetch and return an `SingleChannelImageStack` object and its associated context.

        This method calls the subclass-implemented `_get_payload` method to retrieve the data and context.

        Returns:
            tuple[SingleChannelImageStack, dict]:
                A tuple where the first element is the `SingleChannelImageStack` object and
                the second element is a dictionary representing the context or metadata.
        """
        return self._get_payload(*args, **kwargs)

    def output_data_type(self) -> type[SingleChannelImageStack]:  # type: ignore
        """
        Returns the expected output data type for the data.

        Returns:
            type: `SingleChannelImageStack`, the expected type for the data part of the payload.
        """
        return SingleChannelImageStack
