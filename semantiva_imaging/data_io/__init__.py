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

from .url_loader import UrlLoader
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
    RGBImageStackVideoLoader,
    RGBImageStackAVISaver,
    AnimatedGifRGBImageStackLoader,
    AnimatedGifRGBImageStackSaver,
    # RGBA image stack loaders/savers
    AnimatedGifRGBAImageStackLoader,
    AnimatedGifRGBAImageStackSaver,
)

__all__ = [
    "UrlLoader",
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
    "RGBImageStackVideoLoader",
    "RGBImageStackAVISaver",
    "AnimatedGifRGBImageStackLoader",
    "AnimatedGifRGBImageStackSaver",
    # RGBA image stack loaders/savers
    "AnimatedGifRGBAImageStackLoader",
    "AnimatedGifRGBAImageStackSaver",
]
