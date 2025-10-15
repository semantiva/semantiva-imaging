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

"""Factory for wrapping OpenCV routines as Semantiva operations.

This module provides a factory function that dynamically creates Semantiva DataOperation
classes from OpenCV functions. The factory handles several complex requirements:

1. **Channel Reordering**: OpenCV uses BGR channel order while Semantiva images may use
   RGB or other orderings. The factory automatically creates zero-copy views that
   reorder channels as needed.

2. **Signature Synthesis**: OpenCV functions often have complex signatures that need
   to be parsed from docstrings. The factory provides multiple parsing strategies
   and allows for signature overrides.

3. **Tuple Return Handling**: Many OpenCV functions return tuples (e.g., threshold
   returns both the threshold value and the processed image). The factory uses a
   return_map to route tuple elements to either the final result or observer
   notifications.

4. **Type Safety**: The generated processors validate input types and provide
   proper type annotations for Semantiva's introspection system.

5. **Registry Integration**: The generated classes are properly configured for
   automatic discovery by Semantiva's ProcessorRegistry, including correct
   __module__ attribution for filtering during registration.

The main entry point is `_create_opencv_processor()` which creates a new DataOperation
class that wraps an OpenCV function with all the necessary adaptations.

Example
-------
>>> import cv2
>>> from semantiva_imaging.data_types import RGBImage
>>>
>>> # Create a Gaussian blur processor for RGB images
>>> GaussianBlurProcessor = _create_opencv_processor(
...     cv2.GaussianBlur,
...     "GaussianBlurProcessor",
...     RGBImage,
...     RGBImage
... )
>>>
>>> # Use the processor
>>> processor = GaussianBlurProcessor()
>>> blurred_image = processor.process(rgb_image, ksize=(5, 5), sigmaX=1.0)

Notes
-----
The factory assumes OpenCV functions follow BGR channel ordering for 3+ channel
images. Single channel images are passed through without reordering.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Type, cast
import re
import numpy as np

from semantiva.data_processors import DataOperation
from semantiva.data_types import BaseDataType
from semantiva_imaging.processing.base_nchannel import NChannelImageOperation
from semantiva_imaging.processing.processors import SingleChannelImageOperation
from semantiva_imaging.data_types import NChannelImage


class TypeMismatchError(TypeError):
    """Raised when input image does not match the expected channel order.

    This exception is raised when a generated OpenCV processor receives an input
    image that doesn't match the expected input type. For example, passing an
    RGBAImage to a processor that expects RGBImage.

    This is a subclass of TypeError to maintain compatibility with Python's
    type system expectations.
    """


# OpenCV uses BGR channel ordering for 3+ channel images
# This constant defines the expected channel order for OpenCV functions
_CV_CHANNEL_ORDER = ("B", "G", "R")


def _reorder_view(
    arr: np.ndarray, src: tuple[str, ...], dst: tuple[str, ...]
) -> np.ndarray:
    """Create a zero-copy view of an array with reordered channels.

    This function creates a view of the input array with channels reordered
    from the source ordering to the destination ordering. The operation is
    zero-copy when possible (e.g., for simple RGB<->BGR swaps).

    Parameters
    ----------
    arr : numpy.ndarray
        Input array with shape (..., channels)
    src : tuple[str, ...]
        Source channel ordering (e.g., ("R", "G", "B"))
    dst : tuple[str, ...]
        Destination channel ordering (e.g., ("B", "G", "R"))

    Returns
    -------
    numpy.ndarray
        View of the input array with channels reordered. This is a zero-copy
        operation when possible.

    Notes
    -----
    Common optimizations:
    - If src == dst, returns the original array unchanged
    - For RGB<->BGR swaps, uses efficient slicing: arr[..., ::-1]
    - For other orderings, uses advanced indexing which may create a copy

    Examples
    --------
    >>> rgb_array = np.array([[[255, 0, 0]]])  # Red pixel
    >>> bgr_view = _reorder_view(rgb_array, ("R", "G", "B"), ("B", "G", "R"))
    >>> bgr_view[0, 0]  # [0, 0, 255] - now Blue channel first
    """

    if src == dst:
        return arr
    if src == ("R", "G", "B") and dst == ("B", "G", "R"):
        return arr[..., ::-1]
    if src == ("B", "G", "R") and dst == ("R", "G", "B"):
        return arr[..., ::-1]
    idx = [src.index(ch) for ch in dst]
    return arr[..., idx]


def _notify(op: DataOperation, key: str, value: Any) -> None:
    """Notify observers of a context update through the operation's notification system.

    This function abstracts the notification mechanism used by Semantiva operations
    to communicate auxiliary results (like thresholds, counts, etc.) to observers.
    It handles both the newer _notify_observers method and the legacy
    _notify_context_update method.

    Parameters
    ----------
    op : DataOperation
        The operation instance that should send the notification
    key : str
        The context key for the notification (e.g., "threshold", "contour_count")
    value : Any
        The value to associate with the key

    Notes
    -----
    This function is used internally by the tuple return handling logic to
    route non-image results from OpenCV functions to the appropriate
    notification channels.
    """
    if hasattr(op, "_notify_observers"):
        getattr(op, "_notify_observers")(key, value)
    else:
        op._notify_context_update(key, value)


def _parse_docstring_signature(func: Callable) -> inspect.Signature:
    """Parse a minimal signature from an OpenCV style docstring.

    OpenCV functions often have complex or missing introspection signatures,
    but their docstrings contain signature information in a specific format.
    This function extracts parameter information from the first line of the
    docstring.

    Parameters
    ----------
    func : Callable
        The OpenCV function to parse

    Returns
    -------
    inspect.Signature
        A signature object with parameters extracted from the docstring.
        Optional parameters (marked with [brackets]) get default=None.

    Notes
    -----
    This parser handles OpenCV's docstring format where:
    - The first line contains: "functionName(param1, param2[, optional_param])"
    - Optional parameters are enclosed in square brackets
    - Parameters may be nested in multiple bracket levels

    If no signature is found, returns a minimal signature with just "src".

    Examples
    --------
    For a docstring starting with "GaussianBlur(src, ksize[, sigmaX[, sigmaY]])",
    this would extract parameters: src, ksize, sigmaX (optional), sigmaY (optional)

    Limitations
    -----------
    - Only extracts parameter names, not types or detailed defaults
    - Assumes all optional parameters have default=None
    - May not handle complex nested bracket structures correctly
    """
    doc = func.__doc__ or ""
    first_line = doc.strip().splitlines()[0]
    m = re.search(rf"{func.__name__}\(([^)]*)\)", first_line)
    if not m:
        return inspect.Signature(
            [inspect.Parameter("src", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
    param_str = m.group(1)
    params: list[tuple[str, bool]] = []
    optional = False
    cur = ""
    for ch in param_str:
        if ch == "[":
            if cur.strip():
                params.append((cur.strip(), optional))
                cur = ""
            optional = True
            continue
        if ch == "]":
            if cur.strip():
                params.append((cur.strip(), optional))
                cur = ""
            continue
        if ch == ",":
            if cur.strip():
                params.append((cur.strip(), optional))
                cur = ""
            continue
        cur += ch
    if cur.strip():
        params.append((cur.strip(), optional))

    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None if opt else inspect._empty,
        )
        for name, opt in params
    ]
    return inspect.Signature(parameters=parameters)


def _simple_parser(func: Callable) -> inspect.Signature:
    """Parse function signature using standard introspection with docstring fallback.

    This is the default parser for most OpenCV functions. It first attempts
    to use Python's standard inspect.signature() and falls back to docstring
    parsing if that fails.

    Parameters
    ----------
    func : Callable
        The OpenCV function to parse

    Returns
    -------
    inspect.Signature
        Function signature extracted via introspection or docstring parsing

    Notes
    -----
    This parser is suitable for most OpenCV functions that have either:
    - Proper Python signatures available via introspection
    - Standard OpenCV docstring format for signature extraction
    """
    try:
        return inspect.signature(func)
    except (ValueError, TypeError):
        return _parse_docstring_signature(func)


def _nested_parser(func: Callable) -> inspect.Signature:
    """Parse function signature for functions with nested parameter structures.

    This parser is intended for OpenCV functions that have complex nested
    parameter structures in their signatures. Currently it uses the same
    logic as _simple_parser but is kept separate for future enhancement.

    Parameters
    ----------
    func : Callable
        The OpenCV function to parse

    Returns
    -------
    inspect.Signature
        Function signature extracted via introspection or docstring parsing

    Notes
    -----
    This parser is currently identical to _simple_parser but is maintained
    as a separate function for future enhancement to handle complex nested
    parameter structures that may require special parsing logic.
    """
    try:
        return inspect.signature(func)
    except (ValueError, TypeError):
        return _parse_docstring_signature(func)


def _multi_return_parser(func: Callable) -> inspect.Signature:
    """Parse function signature for functions with multiple return values.

    This parser is designed for OpenCV functions that return tuples of values.
    It uses the same signature parsing logic as other parsers but is kept
    separate for functions that are known to have tuple returns.

    Parameters
    ----------
    func : Callable
        The OpenCV function to parse

    Returns
    -------
    inspect.Signature
        Function signature extracted via introspection or docstring parsing

    Notes
    -----
    This parser is currently identical to _simple_parser but is maintained
    as a separate function for functions that are known to return tuples.
    The parser selection logic uses docstring analysis to identify such functions.
    """
    try:
        return inspect.signature(func)
    except (ValueError, TypeError):
        return _parse_docstring_signature(func)


def _choose_parser(func: Callable) -> Callable[[Callable], inspect.Signature]:
    """Choose the appropriate signature parser based on function characteristics.

    This function analyzes the OpenCV function's docstring to determine which
    signature parser would be most appropriate. It looks for specific patterns
    in the docstring that indicate the function's complexity.

    Parameters
    ----------
    func : Callable
        The OpenCV function to analyze

    Returns
    -------
    Callable[[Callable], inspect.Signature]
        The most appropriate parser function for this OpenCV function

    Notes
    -----
    Parser selection logic:
    - If the docstring contains "->" with comma-separated return types,
      chooses _multi_return_parser for tuple return handling
    - Otherwise, chooses _simple_parser for standard function parsing

    Future enhancements could analyze other docstring patterns to select
    _nested_parser for functions with complex parameter structures.
    """
    doc = func.__doc__ or ""
    first_line = doc.splitlines()[0] if doc.splitlines() else ""
    if "->" in first_line:
        parts = first_line.split("->", 1)
        if len(parts) > 1 and "," in parts[1]:
            return _multi_return_parser
    return _simple_parser


# Public factory -----------------------------------------------------------------


def _create_opencv_processor(
    cv_func: Callable,
    name: str,
    input_type: Type[BaseDataType],
    output_type: Type[BaseDataType],
    *,
    signature_parser: Optional[Callable[[Callable], inspect.Signature]] = None,
    override_signature: Optional[inspect.Signature] = None,
    return_map: Optional[Dict[int, str]] = None,
) -> Type[DataOperation]:
    """Create a Semantiva DataOperation that wraps an OpenCV function.

    This factory function dynamically generates a new DataOperation class that
    wraps an OpenCV function with automatic channel reordering, signature synthesis,
    and tuple return handling. The generated class integrates seamlessly with
    Semantiva's pipeline system.

    Parameters
    ----------
    cv_func : Callable
        The OpenCV function to wrap (e.g., cv2.GaussianBlur, cv2.threshold)
    name : str
        Name for the generated processor class (e.g., "GaussianBlurProcessor")
    input_type : Type[Any]
        Expected input image type (e.g., RGBImage, SingleChannelImage)
    output_type : Type[Any]
        Expected output image type (e.g., RGBImage, SingleChannelImage)
    signature_parser : Optional[Callable[[Callable], inspect.Signature]], optional
        Custom function to parse the OpenCV function's signature. If None,
        an appropriate parser is chosen automatically based on the function's
        docstring characteristics.
    override_signature : Optional[inspect.Signature], optional
        If provided, this signature is used directly instead of parsing the
        OpenCV function. Useful for functions with problematic signatures.
    return_map : Optional[Dict[int, str]], optional
        Maps tuple return indices to context keys for observer notifications.
        For example, {0: "threshold"} would send the first tuple element
        to observers with key "threshold".

    Returns
    -------
    Type[DataOperation]
        A new DataOperation class that wraps the OpenCV function with:
        - Automatic channel reordering between Semantiva and OpenCV formats
        - Proper signature for Semantiva's introspection system
        - Type validation for input images
        - Tuple return handling with observer notifications

    Examples
    --------
    Basic usage with a simple OpenCV function:

    >>> import cv2
    >>> from semantiva_imaging.data_types import RGBImage
    >>>
    >>> BlurProcessor = _create_opencv_processor(
    ...     cv2.GaussianBlur, "BlurProcessor", RGBImage, RGBImage
    ... )
    >>> processor = BlurProcessor()
    >>> result = processor.process(rgb_image, ksize=(5, 5), sigmaX=1.0)

    Advanced usage with tuple returns and custom signature:

    >>> from semantiva_imaging.data_types import SingleChannelImage
    >>>
    >>> # cv2.threshold returns (threshold_value, thresholded_image)
    >>> ThresholdProcessor = _create_opencv_processor(
    ...     cv2.threshold,
    ...     "ThresholdProcessor",
    ...     SingleChannelImage,
    ...     SingleChannelImage,
    ...     return_map={0: "threshold_value"}  # Send threshold to observers
    ... )
    >>> processor = ThresholdProcessor()
    >>> # The threshold value will be sent to observers, result is the image
    >>> result = processor.process(image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    Notes
    -----
    Channel Reordering:
    The factory automatically handles channel reordering between Semantiva image
    formats (which may use RGB, RGBA, etc.) and OpenCV's BGR format. This is
    done with zero-copy views when possible.

    Signature Synthesis:
    OpenCV functions often have complex signatures that need special handling.
    The factory provides multiple parsing strategies and allows for complete
    signature overrides when needed.

    Tuple Return Handling:
    Many OpenCV functions return tuples (e.g., cv2.threshold returns both the
    threshold value and the processed image). The return_map parameter allows
    routing these values to appropriate destinations.

    Type Safety:
    The generated processors validate input types at runtime and provide proper
    type annotations for Semantiva's introspection system.

    Raises
    ------
    TypeMismatchError
        When the input image type doesn't match the expected input_type
    ValueError
        When tuple return handling fails (e.g., no image payload found)
    """

    # Step 1: Determine the signature parsing strategy
    # If no custom parser or override is provided, choose based on function characteristics
    parser = _choose_parser(cv_func) if signature_parser is None else signature_parser

    # Step 2: Build the signature for the generated processor
    if override_signature is not None:
        # Use the provided signature directly
        new_sig = override_signature
    else:
        # Parse the OpenCV function's signature and adapt it for Semantiva
        sig = parser(cv_func)
        params = list(sig.parameters.values())[
            1:
        ]  # Skip the first parameter (src/input)
        new_params = [
            # Add 'self' parameter for the method
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            # Replace the first parameter with our typed 'data' parameter
            inspect.Parameter(
                "data", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=input_type
            ),
            # Keep all other parameters unchanged
            *params,
        ]
        new_sig = inspect.Signature(
            parameters=new_params, return_annotation=output_type
        )

    # Step 3: Set up return mapping for tuple handling
    return_map = return_map or {}

    def _process_logic(self, data: BaseDataType, *args, **kwargs) -> BaseDataType:
        """Process an image through the wrapped OpenCV function.

        This method handles the complete pipeline:
        1. Type validation of input image
        2. Channel reordering for OpenCV compatibility
        3. OpenCV function invocation
        4. Tuple return processing and observer notifications
        5. Channel reordering back to Semantiva format
        6. Result wrapping in output type

        The method signature is dynamically set to match the OpenCV function's
        parameters with appropriate type annotations.
        """
        # Step 3a: Validate input type
        if not isinstance(data, input_type):
            raise TypeMismatchError(
                f"Expected {input_type.__name__}, got {type(data).__name__}"
            )

        # Step 3b: Extract the underlying array data
        arr: np.ndarray = getattr(data, "data")

        # Step 3c: Create a view with OpenCV-compatible channel ordering
        # Only reorder if we have 3+ channels (assuming RGB-like data)
        # Ensure that channel reordering is only applied to multi-channel images.
        if hasattr(data, "channel_info") and len(getattr(data, "channel_info")) >= 3:
            view: np.ndarray = _reorder_view(
                arr, tuple(data.channel_info), _CV_CHANNEL_ORDER
            )
        else:
            # Single channel or channel-agnostic data passes through unchanged
            view = arr

        # Step 3d: Call the OpenCV function with the reordered view
        result = cv_func(view, *args, **kwargs)

        # Step 3e: Handle tuple returns with observer notifications
        if isinstance(result, tuple):
            unmatched = []  # Collect tuple elements not mapped to observers

            # Process each element in the tuple
            for idx, val in enumerate(result):
                if idx in return_map:
                    # Send this element to observers with the mapped key
                    _notify(self, return_map[idx], val)
                else:
                    # This element is not mapped, could be the image result
                    unmatched.append(val)

            # Determine which element is the actual image result
            if not unmatched:
                raise ValueError("No image payload returned from OpenCV call")
            if len(unmatched) == 1:
                # Only one unmapped element, use it as the result
                result = unmatched[0]
            elif return_map:
                # Multiple unmapped elements but we have a return_map,
                # assume the last one is the image result
                result = unmatched[-1]
            else:
                # Multiple unmapped elements and no return_map guidance
                raise ValueError("Ambiguous tuple return without return_map")

        # Step 3f: Reorder channels back to Semantiva format
        if (
            isinstance(result, np.ndarray)
            and result.ndim >= 3
            and result.shape[-1] >= 3
            and hasattr(data, "channel_info")
            and len(getattr(data, "channel_info")) >= 3
        ):
            # Convert back from OpenCV BGR to the original channel ordering
            result = _reorder_view(result, _CV_CHANNEL_ORDER, tuple(data.channel_info))

        # Step 3g: Wrap result in the expected output type
        return cast(Any, output_type)(result)

    # Step 4: Define the class attributes for the generated processor
    attrs = {
        "__doc__": f'Semantiva wrapper for "{cv_func.__name__}"\n'
        + (cv_func.__doc__ or ""),
        "__module__": __name__,
        "_process_logic": _process_logic,
        "input_data_type": classmethod(lambda cls: input_type),
        "output_data_type": classmethod(lambda cls: output_type),
    }

    # Step 4a: Add context keys if return_map is provided
    if return_map:
        # Register the context keys from return_map
        context_keys = list(return_map.values())
        attrs["context_keys"] = classmethod(lambda cls: context_keys)

    # Step 5: Choose the appropriate base class
    # Use NChannelImageOperation for multi-channel images, SingleChannelImageOperation otherwise
    base_cls: Type[DataOperation]
    if issubclass(input_type, NChannelImage):
        base_cls = cast(Type[DataOperation], NChannelImageOperation)
    else:
        base_cls = cast(Type[DataOperation], SingleChannelImageOperation)

    # Step 6: Dynamically create the new processor class
    generated_cls = type(name, (base_cls,), attrs)

    # Step 6a: Set the __module__ attribute to the caller's module for proper registry detection
    # The ProcessorRegistry filters classes by __module__ to ensure they belong to the
    # module being registered. Since dynamically created classes don't get the proper
    # __module__ set by default, we need to explicitly set it to the caller's module.
    current_frame = inspect.currentframe()
    if current_frame is not None:
        caller_frame = current_frame.f_back
        if caller_frame and caller_frame.f_globals.get("__name__"):
            generated_cls.__module__ = caller_frame.f_globals["__name__"]

    # Step 7: Attach the synthesized signature to the _process_logic method
    # This enables Semantiva's introspection system to discover the correct parameters
    cast(Any, getattr(generated_cls, "_process_logic")).__signature__ = new_sig

    return generated_cls
