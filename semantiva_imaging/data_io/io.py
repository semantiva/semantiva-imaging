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

from typing_extensions import no_type_check
from semantiva.data_io import DataSource, PayloadSource, DataSink, PayloadSink
from ..data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
    RGBImage,
    RGBImageStack,
    RGBAImage,
    RGBAImageStack,
    MatplotlibFigure,
)


class MatplotlibFigureDataSource(DataSource):
    """Abstract base class for matplotlib figure data sources."""

    @classmethod
    def output_data_type(cls) -> type[MatplotlibFigure]:
        """Returns the expected output data type: MatplotlibFigure."""
        return MatplotlibFigure


class SingleChannelImageDataSource(DataSource):
    """
    Abstract base class for image data sources.
    """

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImage]:
        """Returns the expected output data type for SingleChannelImageDataSource: SingleChannelImage."""
        return SingleChannelImage


class SingleChannelImageStackSource(DataSource):
    """
    Abstract base class for image stack data sources.
    """

    @classmethod
    def get_data(cls, *args, **kwargs):
        """
        Fetch and return `SingleChannelImageStack` data.

        This stateless classmethod calls the subclass-implemented `_get_data`
        classmethod to retrieve the data.

        Returns:
            SingleChannelImageStack: The fetched image stack data.
        """
        return cls._get_data(*args, **kwargs)

    @classmethod
    def output_data_type(cls) -> type[SingleChannelImageStack]:  # type: ignore
        """Returns the expected output data type for SingleChannelImageStackSource: SingleChannelImageStack."""
        return SingleChannelImageStack


class SingleChannelImageDataSink(DataSink):
    """
    Abstract base class for image data sinks.
    """

    @classmethod
    def input_data_type(cls):  # type: ignore
        """Returns the expected input data type for SingleChannelImageDataSink: SingleChannelImage."""
        return SingleChannelImage


class SingleChannelImageStackSink(DataSink):
    """
    Abstract base class for SingleChannelImageStack sinks.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImageStack]:  # type: ignore
        """Returns the expected input data type for SingleChannelImageStack."""
        return SingleChannelImageStack


class RGBImageDataSource(DataSource):
    """
    Abstract base class for RGB image data sources. # type: ignore

    """

    @classmethod
    def output_data_type(cls) -> type[RGBImage]:
        """Returns the expected output data type for RGBImageDataSource: RGBImage."""
        return RGBImage


class RGBImageStackSource(DataSource):
    """
    Abstract base class for RGB image stack data sources.
    """

    @classmethod
    def output_data_type(cls) -> type[RGBImageStack]:
        """Returns the expected output data type for RGBImageStackSource: RGBImageStack."""
        return RGBImageStack


class RGBAImageDataSource(DataSource):
    """
    Abstract base class for RGBA image data sources.
    """

    @classmethod
    def output_data_type(cls) -> type[RGBAImage]:
        """Returns the expected output data type for RGBAImageDataSource: RGBAImage."""
        return RGBAImage


class RGBAImageStackSource(DataSource):
    """
    Abstract base class for RGBA image stack data sources.
    """

    @classmethod
    def output_data_type(cls) -> type[RGBAImageStack]:
        """Returns the expected output data type for RGBAImageStackSource: RGBAImageStack."""
        return RGBAImageStack


class RGBImageDataSink(DataSink):
    """
    Abstract base class for RGB image data sinks.
    """

    @classmethod
    @no_type_check
    def input_data_type(cls) -> type[RGBImage]:
        """Returns the expected input data type for RGBImageDataSink: RGBImage."""
        return RGBImage


class RGBImageStackSink(DataSink):
    """
    Abstract base class for RGB image stack sinks.
    """

    @classmethod
    @no_type_check
    def input_data_type(cls) -> type[RGBImageStack]:
        """Returns the expected input data type for RGBImageStackSink: RGBImageStack."""
        return RGBImageStack


class RGBAImageDataSink(DataSink):
    """
    Abstract base class for RGBA image data sinks.
    """

    @classmethod
    @no_type_check
    def input_data_type(cls) -> type[RGBAImage]:
        """Returns the expected input data type for RGBAImageDataSink: RGBAImage."""
        return RGBAImage


class RGBAImageStackSink(DataSink):
    """
    Abstract base class for RGBA image stack sinks.
    """

    @classmethod
    @no_type_check
    def input_data_type(cls) -> type[RGBAImageStack]:
        """Returns the expected input data type for RGBAImageStackSink: RGBAImageStack."""
        return RGBAImageStack


class ImagePayloadSink(PayloadSink):
    """Abstract base class for sinks that handle ``SingleChannelImage`` payloads.

    Sinks implementing this interface consume image data together with its
    contextual information and persist them using a backend specific
    mechanism.
    """

    @classmethod
    def input_data_type(cls) -> type[SingleChannelImage]:  # type: ignore
        """
        Returns the expected input data type for the data.

        Returns:
            type: `SingleChannelImage`, the expected type for the data parameter.
        """
        return SingleChannelImage


class SingleChannelImageStackPayloadSource(PayloadSource):
    """Source of ``SingleChannelImageStack`` objects with context information."""

    @classmethod
    def output_data_type(self):
        """
        Returns the expected output data type for the data.

        Returns:
            type: `SingleChannelImageStack`, the expected type for the data part of the payload.
        """
        return SingleChannelImageStack

    @classmethod
    def injected_context_keys(cls) -> list[str]:
        return []
