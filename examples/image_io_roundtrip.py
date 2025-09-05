"""Demonstrate basic image and video round-trip saving/loading."""

import numpy as np
from semantiva_imaging.data_types import (
    SingleChannelImage,
    SingleChannelImageStack,
)
from semantiva_imaging.data_io.loaders_savers import (
    JpgSingleChannelImageSaver,
    JpgSingleChannelImageLoader,
    PngSingleChannelImageSaver,
    PngSingleChannelImageLoader,
    TiffSingleChannelImageSaver,
    TiffSingleChannelImageLoader,
    SingleChannelImageStackAVISaver,
    SingleChannelImageStackVideoLoader,
)


if __name__ == "__main__":
    gray = SingleChannelImage(np.random.randint(0, 255, (5, 5), dtype=np.uint8))

    JpgSingleChannelImageSaver().send_data(gray, "roundtrip.jpg")
    print(JpgSingleChannelImageLoader().get_data("roundtrip.jpg"))

    PngSingleChannelImageSaver().send_data(gray, "roundtrip.png")
    print(PngSingleChannelImageLoader().get_data("roundtrip.png"))

    TiffSingleChannelImageSaver().send_data(gray, "roundtrip.tiff")
    print(TiffSingleChannelImageLoader().get_data("roundtrip.tiff"))

    stack = SingleChannelImageStack(
        np.random.randint(0, 255, (2, 5, 5), dtype=np.uint8)
    )
    SingleChannelImageStackAVISaver().send_data(stack, "roundtrip.avi")
    print(SingleChannelImageStackVideoLoader().get_data("roundtrip.avi"))
