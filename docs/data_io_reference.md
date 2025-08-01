# Data IO Classes

The following loaders and savers are available in `semantiva_imaging.data_io.loaders_savers`:

- `NpzSingleChannelImageLoader` / `NpzSingleChannelImageDataSaver`
- `NpzSingleChannelImageStackDataLoader` / `NpzSingleChannelImageStackDataSaver`
- `PngSingleChannelImageLoader` / `PngSingleChannelImageSaver`
- `JpgSingleChannelImageLoader` / `JpgSingleChannelImageSaver`
- `TiffSingleChannelImageLoader` / `TiffSingleChannelImageSaver`
- `JpgRGBImageLoader` / `JpgRGBImageSaver`
- `PngRGBImageLoader` / `PngRGBImageSaver`
- `TiffRGBImageLoader` / `TiffRGBImageSaver`
- `PngRGBAImageLoader` / `PngRGBAImageSaver`
- `SingleChannelImageStackVideoLoader` / `SingleChannelImageStackAVISaver`
- `RGBImageStackVideoLoader` / `RGBImageStackAVISaver`
- `AnimatedGifRGBAImageStackLoader` / `AnimatedGifSinglechannelImageStackSaver`
- Additionally, HTTP/HTTPS sources can be loaded using `UrlLoader`: `UrlLoader(PngSingleChannelImageLoader)`
