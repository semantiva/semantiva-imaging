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

"""Semantiva Imaging extension package.

This extension provides comprehensive image processing capabilities built on top
of the Semantiva framework. It includes support for various image data types,
processing operations, probes for inspection, and I/O operations.

Usage in YAML:
    extensions: ["semantiva-imaging"]

Components provided:
- Image data types (SingleChannelImage, RGBImage, etc.)
- Image processing operations (filters, transforms, etc.)
- Image I/O operations (loaders and savers)
- OpenCV adapter components
- Image probes for inspection and analysis
"""

from semantiva.registry import SemantivaExtension
from semantiva.registry.processor_registry import ProcessorRegistry


class ImagingExtension(SemantivaExtension):
    """Extension for image processing."""

    def register(self) -> None:
        """Register imaging processors with the ProcessorRegistry."""
        ProcessorRegistry.register_modules(
            [
                "semantiva_imaging.data_types.data_types",
                "semantiva_imaging.data_types.mpl_figure",
                "semantiva_imaging.processing.operations",
                "semantiva_imaging.data_io.parametric_line",
                "semantiva_imaging.data_io.parametric_surface",
                "semantiva_imaging.processing.figure_render",
                "semantiva_imaging.probes.probes",
                "semantiva_imaging.data_io.loaders_savers",
                "semantiva_imaging.adapters.opencv_library.builders",
            ]
        )
