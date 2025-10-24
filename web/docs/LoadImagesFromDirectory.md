# LoadImagesFromDirectory

The LoadImagesFromDirectory node is a powerful batch image loader that efficiently processes multiple images from a directory with advanced resizing, sorting, and formatting options. This node is essential for keyframe-based workflows and batch processing tasks where consistent image dimensions and ordering are critical.

## Overview

This node loads all supported image files from a specified directory, applies consistent resizing, sorts them according to various criteria, and outputs them as a properly formatted batch tensor ready for use in ComfyUI workflows. It's particularly useful for creating keyframe sequences, batch processing workflows, and preparing image datasets.

## Key Features

- **Batch Loading**: Process entire directories of images in one operation
- **Smart Resizing**: Multiple resize modes to handle varying input dimensions
- **Flexible Sorting**: Sort images by name, date, size, or preserve original order
- **Format Support**: Wide range of image formats (PNG, JPG, JPEG, BMP, WEBP, TIF, TIFF)
- **Consistent Output**: All images normalized to identical dimensions and format
- **Reload Control**: Option to force refresh on each execution
- **Error Handling**: Robust error reporting for missing files or invalid formats

## Parameters

### Required Inputs

#### `directory` (STRING)
The directory path containing images to load, relative to ComfyUI's input directory.

**Path Resolution:**
- Path is relative to `ComfyUI/input/` directory
- Use forward slashes or backslashes as appropriate for your system
- Subdirectories are supported (e.g., `keyframes/sequence1`)

**Examples:**
```
"keyframes" → loads from ComfyUI/input/keyframes/
"project/frames" → loads from ComfyUI/input/project/frames/
"batch_001" → loads from ComfyUI/input/batch_001/
```

**Default:** `keyframes`

#### `reload_on_execute` (BOOLEAN)
Forces the node to reload images on every execution, bypassing ComfyUI's caching system.

**When to use:**
- **True**: When images in the directory change frequently during workflow development
- **False**: For stable directories where images don't change (better performance)

**Default:** `False`

#### `target_width` (INT)
Target width in pixels for all output images.

- **Range:** 64 - 8192 pixels
- **Step:** 8 pixels
- **Default:** 512

#### `target_height` (INT) 
Target height in pixels for all output images.

- **Range:** 64 - 8192 pixels
- **Step:** 8 pixels  
- **Default:** 512

#### `resize_mode` (ENUM)
Determines how images are resized to match target dimensions.

**Options:**

##### `stretch`
Directly stretches images to exact target dimensions.
- **Pros**: Exact dimensions guaranteed, simple processing
- **Cons**: May distort aspect ratio, can cause visual artifacts
- **Use case**: When exact dimensions are critical and aspect ratio changes are acceptable

##### `fit`
Maintains original aspect ratio, fits image inside target dimensions with padding.
- **Pros**: Preserves aspect ratio, no cropping
- **Cons**: May have black padding, doesn't fill entire frame
- **Use case**: When preserving entire image content is important
- **Padding**: Black padding added to center the image

##### `crop` (Default)
Maintains original aspect ratio, crops image to fill target dimensions.
- **Pros**: Fills entire frame, preserves aspect ratio
- **Cons**: May lose edge content through cropping
- **Use case**: When filling the frame completely is important
- **Cropping**: Intelligent center cropping to preserve main subject

**Default:** `crop`

#### `sort_mode` (ENUM)
Controls the order in which images are loaded and processed.

**Options:**

##### `name_asc` (Default)
Sort alphabetically by filename in ascending order (A-Z, 0-9).
- **Best for**: Numbered sequences (001.jpg, 002.jpg, etc.)
- **Example order**: frame_001.png, frame_002.png, frame_010.png

##### `name_desc`  
Sort alphabetically by filename in descending order (Z-A, 9-0).
- **Best for**: Reverse sequences or reverse alphabetical order

##### `date_asc`
Sort by file modification date, oldest first.
- **Best for**: Processing images in chronological order of creation/modification
- **Useful for**: Photo sequences, time-lapse workflows

##### `date_desc`
Sort by file modification date, newest first.
- **Best for**: Processing most recent images first

##### `size_asc`
Sort by file size, smallest first.
- **Best for**: Processing lower resolution images before higher resolution
- **Useful for**: Progressive quality workflows

##### `size_desc`
Sort by file size, largest first.
- **Best for**: Processing highest quality images first

##### `none`
No sorting - uses system's default directory listing order.
- **Best for**: When order doesn't matter or preserving system order
- **Note**: Order may vary between operating systems

**Default:** `name_asc`

## Outputs

### `images` (IMAGE)
A batch tensor containing all loaded and processed images.

**Format:** `(batch_size, height, width, 3)`
- **batch_size**: Number of images loaded from directory
- **height**: Target height (as specified in parameters)
- **width**: Target width (as specified in parameters)  
- **3**: RGB color channels

**Data Type:** `torch.FloatTensor`
**Value Range:** 0.0 - 1.0 (normalized from 0-255)

## Supported Image Formats

The node supports the following image formats:
- **PNG** (.png) - Lossless compression, transparency support
- **JPEG** (.jpg, .jpeg) - Lossy compression, smaller file sizes
- **BMP** (.bmp) - Uncompressed bitmap format
- **WebP** (.webp) - Modern web format with good compression
- **TIFF** (.tif, .tiff) - High quality, often used for professional photography

**Note:** All images are converted to RGB format regardless of input format, removing any alpha channels.

## Processing Pipeline

### 1. Directory Scanning
- Scans specified directory for supported image files
- Filters out hidden files (starting with '.')
- Validates file extensions against supported formats

### 2. File Sorting
- Applies selected sort mode to determine processing order
- Handles file system errors gracefully (places problematic files at end)

### 3. Image Loading & Processing
For each image:
- Opens image using PIL (Python Imaging Library)
- Converts to RGB format (removes alpha channels)
- Applies selected resize mode to reach target dimensions
- Converts to ComfyUI-compatible tensor format
- Normalizes values to 0.0-1.0 range

### 4. Batch Assembly
- Concatenates all processed images into a single batch tensor
- Ensures consistent dimensions across all images in batch

## Usage Examples

### Basic Keyframe Loading
```python
# Load numbered keyframes in order
directory = "my_keyframes"
sort_mode = "name_asc"  # Ensures 001.jpg comes before 010.jpg
resize_mode = "crop"    # Fill frame completely
target_width = 512
target_height = 512
```

### Time-lapse Processing
```python
# Process photos chronologically  
directory = "timelapse_photos"
sort_mode = "date_asc"  # Oldest photos first
resize_mode = "fit"     # Preserve entire photo content
target_width = 1920
target_height = 1080
```

### Development Workflow
```python
# Frequently changing images during development
directory = "test_frames"
reload_on_execute = True  # Always check for new/changed images
resize_mode = "stretch"   # Quick processing for testing
target_width = 256        # Lower resolution for speed
target_height = 256
```

### High-Quality Processing
```python
# Process largest images first for quality workflows
directory = "high_res_sources" 
sort_mode = "size_desc"   # Largest files first
resize_mode = "crop"      # Maintain quality, fill frame
target_width = 1024
target_height = 1024
```

## Resize Mode Comparison

| Mode | Aspect Ratio | Full Frame | Content Loss | Use Case |
|------|--------------|------------|--------------|----------|
| **stretch** | May change | ✅ Yes | Distortion | Exact dimensions needed |
| **fit** | ✅ Preserved | ❌ Padded | None | Preserve all content |
| **crop** | ✅ Preserved | ✅ Yes | Edge cropping | Fill frame, natural look |

## Performance Considerations

### Memory Usage
- Memory usage scales with: `batch_size × width × height × 3 × 4 bytes`
- For 100 images at 512×512: ~300MB RAM
- For 100 images at 1024×1024: ~1.2GB RAM

### Processing Speed
- **Fastest**: `stretch` mode (simple resize)
- **Medium**: `crop` mode (crop + resize)  
- **Slowest**: `fit` mode (resize + composition)

### Optimization Tips
- Use `reload_on_execute = False` for stable directories
- Lower target dimensions for faster processing during development
- Consider batch size limits for available system memory
- Use appropriate resize mode for your specific needs

## Error Handling

### Common Errors and Solutions

#### `FileNotFoundError: Directory not found`
**Cause:** Specified directory doesn't exist relative to ComfyUI input folder
**Solution:** 
- Verify directory path is correct
- Ensure directory exists in `ComfyUI/input/`
- Check for typos in directory name

#### `ValueError: No valid images found`
**Cause:** Directory exists but contains no supported image files
**Solution:**
- Verify images are in supported formats
- Check if images are hidden (filename starting with '.')
- Ensure files aren't corrupted

#### `RuntimeError: Failed to load image`
**Cause:** Specific image file is corrupted or in unsupported format
**Solution:**
- Check image file integrity
- Try opening image in standard image viewer
- Remove or replace corrupted files
- Verify file format is supported

#### Memory errors
**Cause:** Too many images or too high resolution for available RAM
**Solution:**
- Reduce target dimensions
- Process fewer images per batch
- Increase system RAM
- Use lower resolution images

## Integration Patterns

### Typical Workflow Connections

#### Keyframe Video Generation
```
LoadImagesFromDirectory → WanKeyframeToVideo
```

#### Batch Processing
```
LoadImagesFromDirectory → Image Processing Node → Save/Export
```

#### Style Transfer Pipeline  
```
LoadImagesFromDirectory → Style Transfer → Batch Save
```

#### Dataset Preparation
```
LoadImagesFromDirectory → Preprocessing → Training Dataset
```

## Best Practices

### Directory Organization
- Use consistent naming conventions (001.jpg, 002.jpg, etc.)
- Keep related images in dedicated subdirectories
- Avoid spaces and special characters in filenames
- Use descriptive directory names

### Parameter Selection
- **For keyframes**: Use `name_asc` sorting with `crop` resize mode
- **For datasets**: Consider `date_asc` for chronological data
- **For development**: Enable `reload_on_execute` temporarily
- **For production**: Disable `reload_on_execute` for better performance

### File Management
- Regularly clean up temporary or test directories
- Keep original high-resolution sources separate from processed versions
- Consider using version control for important image sequences
- Backup critical image directories before batch processing

## Troubleshooting

### Images appear in wrong order
- Check `sort_mode` setting
- Verify filename numbering (use leading zeros: 001 not 1)
- Consider using `date_asc` for chronological ordering

### Images look stretched or distorted
- Switch from `stretch` to `crop` or `fit` mode
- Verify target dimensions match desired aspect ratio
- Check original image aspect ratios

### Memory issues with large batches  
- Reduce `target_width` and `target_height`
- Process images in smaller batches
- Monitor system memory usage
- Consider using image compression

### Inconsistent results between runs
- Enable `reload_on_execute` if images change frequently
- Check for hidden files or temporary files in directory
- Verify all images are in supported formats
