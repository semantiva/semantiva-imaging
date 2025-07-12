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

"""Semantiva Imaging specialization package."""

from semantiva.specializations import SemantivaSpecialization
from semantiva.component_loader import ComponentLoader


class ImagingSpecialization(SemantivaSpecialization):
    """Specialization for image processing."""

    def __init__(self, loader: ComponentLoader | None = None) -> None:
        """Store the loader used for registration."""
        self.loader = loader or ComponentLoader

    def register(self) -> None:
        registered_modules = [
            "semantiva_imaging.processing.operations",
            "semantiva_imaging.probes.probes",
            "semantiva_imaging.data_io.loaders_savers",
            "semantiva_imaging.adapters.opencv_library.builders",
        ]
        ComponentLoader.register_modules(registered_modules)
