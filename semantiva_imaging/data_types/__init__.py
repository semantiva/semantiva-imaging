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

"""Convenience exports for Semantiva imaging data types."""

from .data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
    NChannelImage,
    NChannelImageStack,
    RGBImage,
    RGBImageStack,
    RGBAImage,
    RGBAImageStack,
)
from .mpl_figure import MatplotlibFigure, MatplotlibFigureCollection

__all__ = [
    "SingleChannelImage",
    "SingleChannelImageStack",
    "NChannelImage",
    "NChannelImageStack",
    "RGBImage",
    "RGBImageStack",
    "RGBAImage",
    "RGBAImageStack",
    "MatplotlibFigure",
    "MatplotlibFigureCollection",
]
