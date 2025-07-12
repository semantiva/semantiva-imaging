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

"""I/O interface classes for imaging data."""

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
        raise NotImplementedError

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

    def input_data_type(self) -> type[SingleChannelImageStack]:  # type: ignore
        return SingleChannelImageStack


class ImagePayloadSink(PayloadSink):
    """Abstract base class for sinks that handle ``SingleChannelImage`` payloads.

    Sinks implementing this interface consume image data together with its
    contextual information and persist them using a backend specific
    mechanism.
    """

    @abstractmethod
    @override
    def _send_payload(
        self, data: SingleChannelImage, context: ContextType, *args, **kwargs
    ):
        """Consume ``SingleChannelImage`` data together with its context."""

    def input_data_type(self) -> type[SingleChannelImage]:  # type: ignore
        """
        Returns the expected input data type for the data.

        Returns:
            type: `SingleChannelImage`, the expected type for the data parameter.
        """
        return SingleChannelImage


class SingleChannelImageStackPayloadSource(PayloadSource):
    """Source of ``SingleChannelImageStack`` objects with context information."""

    @abstractmethod
    def _get_payload(
        self, *args, **kwargs
    ) -> tuple[SingleChannelImageStack, ContextType]:
        """Abstract method retrieving an image stack and its context."""

    def output_data_type(self) -> type[SingleChannelImageStack]:  # type: ignore
        """
        Returns the expected output data type for the data.

        Returns:
            type: `SingleChannelImageStack`, the expected type for the data part of the payload.
        """
        return SingleChannelImageStack
