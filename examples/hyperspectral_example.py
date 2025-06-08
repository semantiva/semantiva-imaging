"""
Hyperspectral Example
---------------------
This example demonstrates how to use NChannelImage and NChannelImageStack
for hyperspectral data analysis using BasicImageProbe.

Features:
- Compute band-wise statistics (mean, sum, min, max) using BasicImageProbe.
- Demonstrate the use of SingleChannelImage for individual band analysis.

Dependencies:
- numpy
- semantiva_imaging.data_types
- semantiva_imaging.probes
"""

import numpy as np
from semantiva_imaging.data_types import (
    NChannelImage,
    NChannelImageStack,
    SingleChannelImage,
)
from semantiva_imaging.probes.probes import BasicImageProbe

# Generate a dummy 7-band hyperspectral cube
cube = np.random.rand(4, 4, 7).astype(np.float32)
channel_info = [f"band_{i}" for i in range(7)]

# Create NChannelImage and NChannelImageStack instances
img = NChannelImage(cube, channel_info)
stack = NChannelImageStack(cube[None, ...], channel_info)

# Example Usage
if __name__ == "__main__":
    # Initialize BasicImageProbe for band-wise analysis
    probe = BasicImageProbe()
    print("Per-band statistics via BasicImageProbe:")

    # Iterate through each band and compute statistics
    for idx, name in enumerate(img.channel_info):
        band_arr = img.data[..., idx]  # Extract the band as a 2D array
        single = SingleChannelImage(band_arr)  # Wrap the band in SingleChannelImage
        result = probe.process(single)  # Compute statistics using BasicImageProbe

        # Print the computed statistics for the current band
        print(f"  {name}: {result}")
