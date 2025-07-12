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

"""Factory utilities for creating n-channel data processors."""

from __future__ import annotations
import inspect
from typing import Type, Any, cast

from semantiva.data_processors import DataOperation
from semantiva.data_types import BaseDataType
from ..processing.base_nchannel import NChannelImageOperation
from ..data_types import NChannelImage


def _create_nchannel_processor(
    name: str,
    base_cls: Type[NChannelImageOperation],
    input_type: Type[NChannelImage],
    output_type: Type[NChannelImage],
) -> Type[DataOperation]:
    """Create a public ``DataOperation`` for n-channel image data.

    This factory function dynamically generates a new processor class that wraps
    a base n-channel image operation with specific input/output type constraints.
    The generated class exposes proper semantic metadata (docstring, type info,
    parameter signatures) for Semantiva's core introspection system.

    The factory performs several key transformations:
    1. Creates a new class that inherits from the base operation
    2. Modifies the _process_logic signature to use concrete types
    3. Adds type metadata methods for Semantiva integration
    4. Preserves parameter names, defaults, and annotations
    5. Generates descriptive documentation

    Parameters
    ----------
    name : str
        Name of the generated processor class. This will be the ``__name__``
        attribute of the returned class (e.g., "AddRGBImageProcessor").
    base_cls : Type[DataOperation]
        Base operation class implementing ``_process_logic``. Must be a subclass
        of a class that has a ``_process_logic`` method with the first parameter
        named "data" representing the input data.
    input_type : Type[BaseDataType]
        Concrete input data type expected by the generated processor. The "data"
        parameter in ``_process_logic`` will be annotated with this type.
    output_type : Type[BaseDataType]
        Concrete output data type produced by the generated processor. The
        ``_process_logic`` method's return annotation will be set to this type.

    Returns
    -------
    Type[DataOperation]
        A new processor class that:
        - Inherits from ``base_cls``
        - Has ``input_data_type()`` and ``output_data_type()`` class methods
        - Exposes the correct parameter signature for introspection
        - Includes auto-generated documentation
        - Validates input types at runtime

    Notes
    -----
    The generated class maintains the exact same parameter signature as the
    base class's ``_process_logic`` method, with only the "data" parameter's
    type annotation changed to ``input_type``. All other parameters (names,
    defaults, annotations) are preserved.

    The signature manipulation is necessary because Semantiva's introspection
    system relies on ``inspect.signature()`` to determine processor parameters
    and their metadata. Simply subclassing would inherit generic type annotations
    that don't provide the concrete type information needed for validation.

    Examples
    --------
    >>> class _AddOp(NChannelImageOperation):
    ...     def _process_logic(self, data: NChannelImage, other: NChannelImage, scale: float = 1.0):
    ...         return NChannelImage((data.data + other.data) * scale)
    >>>
    >>> AddRGBProcessor = _create_nchannel_processor(
    ...     "AddRGBProcessor", _AddOp, RGBImage, RGBImage
    ... )
    >>>
    >>> # Generated class has proper metadata
    >>> AddRGBProcessor.input_data_type()  # Returns RGBImage
    >>> AddRGBProcessor.output_data_type()  # Returns RGBImage
    >>> inspect.signature(AddRGBProcessor._process_logic)  # Shows RGBImage annotation
    """
    # Step 1: Extract the original method signature from the base class
    # This captures all parameter information: names, types, defaults, etc.
    sig = inspect.signature(base_cls._process_logic)
    params = list(sig.parameters.values())

    # Step 2: Modify the signature to use concrete input type
    # Replace the "data" parameter's annotation with the specific input_type
    # while preserving all other parameter metadata (defaults, kinds, etc.)
    new_params = [
        p.replace(annotation=input_type) if p.name == "data" else p for p in params
    ]

    # Step 3: Create the new signature with concrete types
    # This new signature will be used for introspection and validation
    new_sig = sig.replace(parameters=new_params, return_annotation=output_type)

    # Step 4: Define metadata methods for Semantiva integration
    # These class methods expose type information to the framework
    def _input_data_type(cls) -> Type[BaseDataType]:
        """Return the concrete input data type for this processor."""
        return input_type

    def _output_data_type(cls) -> Type[BaseDataType]:
        """Return the concrete output data type for this processor."""
        return output_type

    # Step 5: Create the wrapper _process_logic method
    # This method handles signature binding and delegates to the base implementation
    def _process_logic(self, *args, **kwargs):
        """Process data using the base class logic with type-specific signature.

        This wrapper method:
        1. Binds arguments to the concrete signature for validation
        2. Applies parameter defaults from the original signature
        3. Delegates to the base class implementation
        4. Returns the result (base class handles type conversion)
        """
        # Bind arguments to the concrete signature
        # This validates argument count and applies defaults
        bound = new_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Delegate to the base class implementation
        # The base class _process_logic expects the original signature
        result = base_cls._process_logic(**bound.arguments)
        return result

    # Step 6: Attach the concrete signature to the wrapper method
    # This allows inspect.signature() to see the concrete types
    # The cast is necessary because __signature__ is not normally writable
    cast(Any, _process_logic).__signature__ = new_sig

    # Step 7: Define the class attributes for the generated class
    # These attributes define the behavior and metadata of the new class
    attrs = {
        "__doc__": (
            f"Factory-adapted operator for {base_cls.__name__} to ingest {input_type.__name__} "
            f"and produce {output_type.__name__}.\n" + (base_cls.__doc__ or "")
        ),
        "__module__": __name__,  # Set module for proper introspection
        "input_data_type": classmethod(_input_data_type),
        "output_data_type": classmethod(_output_data_type),
        "_process_logic": _process_logic,
    }

    # Step 8: Dynamically create the new class
    # Uses type() to create a class that inherits from base_cls
    # The name parameter becomes the class name
    generated_cls = type(name, (base_cls,), attrs)

    return generated_cls
