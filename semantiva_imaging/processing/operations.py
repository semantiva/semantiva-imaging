import numpy as np
from ..data_types import (
    ImageDataType,
    ImageStackDataType,
)
from ..processing.processors import (
    ImageOperation,
    ImageStackToImageProjector,
)


class ImageSubtraction(ImageOperation):
    """
    Substracts one image from another.
    """

    def _process_logic(
        self, data: ImageDataType, image_to_subtract: ImageDataType
    ) -> ImageDataType:
        """
        Subtracts one image from another.

        Parameters:
            data (ImageDataType): The original image data.
            image_to_subtract (ImageDataType): The image data to subtract.

        Returns:
            ImageDataType: The result of the subtraction operation.
        """
        return ImageDataType(np.subtract(data.data, image_to_subtract.data))


class ImageAddition(ImageOperation):
    """
    Adds two images together.
    """

    def _process_logic(
        self, data: ImageDataType, image_to_add: ImageDataType
    ) -> ImageDataType:
        """
        Adds one image to another.

        Parameters:
            data (ImageDataType): The original image data.
            image_to_add (ImageDataType): The image data to add.

        Returns:
            ImageDataType: The result of the addition operation.
        """
        return ImageDataType(np.add(data.data, image_to_add.data))


class ImageCropper(ImageOperation):
    """
    Crops a rectangular region from the input image.
    """

    def _process_logic(
        self,
        data: ImageDataType,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> ImageDataType:
        """
        Crop a rectangular region from the input image.

        Parameters:
            data (ImageDataType): The original image data.
            x_start (int): The starting x-coordinate of the cropped region.
            x_end (int): The ending x-coordinate of the cropped region.
            y_start (int): The starting y-coordinate of the cropped region.
            y_end (int): The ending y-coordinate of the cropped region.

        Returns:
            ImageDataType: The cropped region of the image.

        Raises:
            ValueError: If the specified cropped region is out of bounds.
        """

        # Ensure the region is within bounds
        if not 0 <= x_start < x_end <= data.data.shape[1]:
            raise ValueError(
                f"x-coordinates out of bounds: x_start={x_start}, x_end={x_end}, width={data.data.shape[1]}"
            )
        if not 0 <= y_start < y_end <= data.data.shape[0]:
            raise ValueError(
                f"y-coordinates out of bounds: y_start={y_start}, y_end={y_end}, height={data.data.shape[0]}"
            )

        cropped_array = data.data[y_start:y_end, x_start:x_end]
        return ImageDataType(cropped_array)


class StackToImageMeanProjector(ImageStackToImageProjector):
    """
    Projects a stack of images into a single image by computing the mean pixel value among the slices.
    """

    def _process_logic(self, data: ImageStackDataType) -> ImageDataType:
        """
        Computes the mean projection of an image stack.

        Parameters:
            data (ImageStackDataType): The input image stack, represented as a 3D NumPy array
                                        (stack of 2D images).

        Returns:
            ImageDataType: A single 2D image resulting from the mean projection of the stack.

        Raises:
            ValueError: If the input data is not a 3D NumPy array.

        """
        # Compute the mean along the stack (first axis)
        return ImageDataType(np.mean(data.data, axis=0))


class ImageNormalizerOperation(ImageOperation):
    """
    Linear normalization of an image data to a specified range.

    """

    def _process_logic(
        self, data: ImageDataType, min_value: float, max_value: float, *args, **kwargs
    ) -> ImageDataType:
        """
        Perform normalization on the image data to scale its pixel values
        linearly to the specified range `[min_value, max_value]`.

        Parameters:
            data (ImageDataType): The image data to normalize.
            min_value (float): The minimum value of the target range.
            max_value (float): The maximum value of the target range.
            *args, **kwargs: Additional parameters for compatibility.

        Returns:
            ImageDataType: The normalized image data with values in `[min_value, max_value]`.
        """

        image_array = data.data
        arr_min = np.min(image_array)
        arr_max = np.max(image_array)

        # Avoid division by zero in case all values in the array are the same
        if arr_max - arr_min == 0:
            return ImageDataType(np.full_like(image_array, (min_value + max_value) / 2))

        # Linear scaling
        scaled_arr = min_value + (image_array - arr_min) * (max_value - min_value) / (
            arr_max - arr_min
        )
        return ImageDataType(scaled_arr)


class ImageStackToSideBySideProjector(ImageStackToImageProjector):
    """
    Projects a stack of images into a single image by concatenating the images side by side.
    """

    def _process_logic(self, data: ImageStackDataType) -> ImageDataType:
        """
        Concatenates the images in the stack side by side.

        Parameters:
            data (ImageStackDataType): The input image stack, represented as a 3D NumPy array
                                        (stack of 2D images).

        Returns:
            ImageDataType: A single 2D image resulting from the concatenation of images in the stack.

        Raises:
            ValueError: If the input data is not a 3D NumPy array.

        """
        # Concatenate the images along the horizontal axis (axis=1)
        return ImageDataType((np.hstack(tuple(data.data))))
