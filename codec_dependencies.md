# Codec-Dependent Classes

## Important Note
Some loader/saver classes in Semantiva Imaging depend on system-specific codecs, which may not be available or consistent across all environments. These classes include:

- **SingleChannelImageStackAVISaver** (AVI)
- **RGBImageStackAVISaver** (AVI)

### System Dependencies
- These classes rely on OpenCV's video codec support.
- Required codecs include MJPG, XVID, mp4v, or X264.
- Availability of these codecs depends on the system's configuration and installed libraries.

### Risks
- **Untested Methods**: These classes are not fully tested due to the lack of codec support on some systems.
- **Potential Failures**: Using these classes on systems without the required codecs will result in runtime errors.
- **Inconsistent Behavior**: Even when codecs are available, behavior may vary across platforms.

### Recommendations
- **Fallback Mechanisms**: Use alternative formats (e.g., Animated GIF) when codec support is unavailable.
- **Error Handling**: Implement robust error handling to gracefully handle missing codecs.
- **System Checks**: Verify codec availability before using these classes.
