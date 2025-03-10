from semantiva.specializations import SemantivaSpecialization
from semantiva.component_loader import ComponentLoader


class ImagingSpecialization(SemantivaSpecialization):
    """Specialization for image processing"""

    def register(self) -> None:
        registered_modules = [
            "semantiva_imaging.processing.operations",
            "semantiva_imaging.probes.probes",
            "semantiva_imaging.data_io.loaders_savers",
        ]
        ComponentLoader.register_modules(registered_modules)
