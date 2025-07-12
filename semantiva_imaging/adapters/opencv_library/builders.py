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

"""Factory-generated OpenCV processors."""

from __future__ import annotations

import cv2

# OpenCV stubs do not define all members used here
# pylint: disable=no-member

from ..opencv_factory import _create_opencv_processor
from ...data_types import RGBImage, SingleChannelImage

# Filters
GaussianBlurRGBImageProcessor = _create_opencv_processor(
    cv2.GaussianBlur, "GaussianBlurRGBImageProcessor", RGBImage, RGBImage
)
MedianBlurRGBImageProcessor = _create_opencv_processor(
    cv2.medianBlur, "MedianBlurRGBImageProcessor", RGBImage, RGBImage
)
BilateralFilterRGBImageProcessor = _create_opencv_processor(
    cv2.bilateralFilter, "BilateralFilterRGBImageProcessor", RGBImage, RGBImage
)

# Color space conversion
RGB2SingleChannelImageProcessor = _create_opencv_processor(
    cv2.cvtColor, "RGB2SingleChannelImageProcessor", RGBImage, SingleChannelImage
)
SingleChannelImageThresholdProcessor = _create_opencv_processor(
    cv2.threshold,
    "SingleChannelImageThresholdProcessor",
    SingleChannelImage,
    SingleChannelImage,
    return_map={0: "threshold_value"},
)

# Edges
CannyEdgeSingleChannelImageProcessor = _create_opencv_processor(
    cv2.Canny,
    "CannyEdgeSingleChannelImageProcessor",
    SingleChannelImage,
    SingleChannelImage,
)
SobelEdgeSingleChannelImageProcessor = _create_opencv_processor(
    cv2.Sobel,
    "SobelEdgeSingleChannelImageProcessor",
    SingleChannelImage,
    SingleChannelImage,
)
LaplacianSingleChannelImageProcessor = _create_opencv_processor(
    cv2.Laplacian,
    "LaplacianSingleChannelImageProcessor",
    SingleChannelImage,
    SingleChannelImage,
)

# Morphology
DilateSingleChannelImageProcessor = _create_opencv_processor(
    cv2.dilate,
    "DilateSingleChannelImageProcessor",
    SingleChannelImage,
    SingleChannelImage,
)
ErodeSingleChannelImageProcessor = _create_opencv_processor(
    cv2.erode,
    "ErodeSingleChannelImageProcessor",
    SingleChannelImage,
    SingleChannelImage,
)

# Rotation helper


def _rotate(src, angle: float, scale: float = 1.0):
    """Rotate an image by a given angle with optional scaling.

    Parameters
    ----------
    src : numpy.ndarray
        Input image array
    angle : float
        Rotation angle in degrees (positive values mean counter-clockwise rotation)
    scale : float, optional
        Scaling factor, by default 1.0

    Returns
    -------
    numpy.ndarray
        Rotated and optionally scaled image
    """
    center = (src.shape[1] // 2, src.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(src, matrix, (src.shape[1], src.shape[0]))


# Transforms
ResizeRGBImageProcessor = _create_opencv_processor(
    cv2.resize, "ResizeRGBImageProcessor", RGBImage, RGBImage
)
RotateRGBImageProcessor = _create_opencv_processor(
    _rotate, "RotateRGBImageProcessor", RGBImage, RGBImage
)
FlipRGBImageProcessor = _create_opencv_processor(
    cv2.flip, "FlipRGBImageProcessor", RGBImage, RGBImage
)
