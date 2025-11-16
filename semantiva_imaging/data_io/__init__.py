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

"""Data I/O module for the Semantiva Imaging extension.

This module provides comprehensive data loading and saving capabilities for various
image formats and types, including support for single-channel, RGB, and RGBA images,
as well as image stacks and video formats.
"""

from .url_loader import UrlLoader
from .parametric_line import ParametricLinePlotGenerator
from .parametric_surface import ParametricSurfacePlotGenerator
from .loaders_savers import (
    # Single channel image loaders/savers
    NpzSingleChannelImageLoader,
    NpzSingleChannelImageDataSaver,
    PngSingleChannelImageLoader,
    PngSingleChannelImageSaver,
    JpgSingleChannelImageLoader,
    JpgSingleChannelImageSaver,
    TiffSingleChannelImageLoader,
    TiffSingleChannelImageSaver,
    # RGB image loaders/savers
    JpgRGBImageLoader,
    JpgRGBImageSaver,
    PngRGBImageLoader,
    PngRGBImageSaver,
    TiffRGBImageLoader,
    TiffRGBImageSaver,
    # RGBA image loaders/savers
    PngRGBAImageLoader,
    PngRGBAImageSaver,
    TiffRGBAImageLoader,
    TiffRGBAImageSaver,
    # Single channel image stack loaders/savers
    NpzSingleChannelImageStackDataLoader,
    NpzSingleChannelImageStackDataSaver,
    PNGSingleChannelImageStackSaver,
    SingleChannelImageStackVideoLoader,
    SingleChannelImageStackAVISaver,
    AnimatedGifSingleChannelImageStackLoader,
    AnimatedGifSingleChannelImageStackSaver,
    # RGB image stack loaders/savers
    PNGRGBImageStackSaver,
    RGBImageStackVideoLoader,
    RGBImageStackAVISaver,
    AnimatedGifRGBImageStackLoader,
    AnimatedGifRGBImageStackSaver,
    # RGBA image stack loaders/savers
    PNGRGBAImageStackSaver,
    AnimatedGifRGBAImageStackLoader,
    AnimatedGifRGBAImageStackSaver,
)

__all__ = [
    "UrlLoader",
    "ParametricLinePlotGenerator",
    "ParametricSurfacePlotGenerator",
    # Single channel image loaders/savers
    "NpzSingleChannelImageLoader",
    "NpzSingleChannelImageDataSaver",
    "PngSingleChannelImageLoader",
    "PngSingleChannelImageSaver",
    "JpgSingleChannelImageLoader",
    "JpgSingleChannelImageSaver",
    "TiffSingleChannelImageLoader",
    "TiffSingleChannelImageSaver",
    # RGB image loaders/savers
    "JpgRGBImageLoader",
    "JpgRGBImageSaver",
    "PngRGBImageLoader",
    "PngRGBImageSaver",
    "TiffRGBImageLoader",
    "TiffRGBImageSaver",
    # RGBA image loaders/savers
    "PngRGBAImageLoader",
    "PngRGBAImageSaver",
    "TiffRGBAImageLoader",
    "TiffRGBAImageSaver",
    # Single channel image stack loaders/savers
    "NpzSingleChannelImageStackDataLoader",
    "NpzSingleChannelImageStackDataSaver",
    "PNGSingleChannelImageStackSaver",
    "SingleChannelImageStackVideoLoader",
    "SingleChannelImageStackAVISaver",
    "AnimatedGifSingleChannelImageStackLoader",
    "AnimatedGifSingleChannelImageStackSaver",
    # RGB image stack loaders/savers
    "PNGRGBImageStackSaver",
    "RGBImageStackVideoLoader",
    "RGBImageStackAVISaver",
    "AnimatedGifRGBImageStackLoader",
    "AnimatedGifRGBImageStackSaver",
    # RGBA image stack loaders/savers
    "PNGRGBAImageStackSaver",
    "AnimatedGifRGBAImageStackLoader",
    "AnimatedGifRGBAImageStackSaver",
]
