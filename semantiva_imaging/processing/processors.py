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

"""Base processor classes for image operations and probes."""

from semantiva.data_processors import DataOperation, DataProbe
from ..data_types import SingleChannelImage, SingleChannelImageStack


class SingleChannelImageOperation(DataOperation):
    """A ``DataOperation`` specialized for :class:`SingleChannelImage` data."""

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


class SingleChannelImageProbe(DataProbe):
    """A ``DataProbe`` for :class:`SingleChannelImage` data."""

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
