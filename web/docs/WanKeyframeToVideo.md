# WanKeyframeToVideo

The WanKeyframeToVideo node is an advanced keyframe-to-video encoder that creates smooth video sequences with intelligent conditioning and latent space interpolation. This node processes multiple keyframe pairs simultaneously to generate coherent video segments with segment-specific prompt conditioning.

## Overview

This node takes a sequence of keyframe images and generates video conditioning data that can be used with video generation models. It creates smooth transitions between keyframes while allowing different prompts to be applied to different segments of the video sequence.

## Key Features

- **Multi-Segment Processing**: Processes multiple keyframe pairs simultaneously for batch video generation
- **Indexed Prompt System**: Apply different prompts to different video segments using `[N]` indexing
- **CLIP Vision Integration**: Enhanced conditioning with optional vision embeddings
- **Flexible Video Control**: Precise control over FPS, duration, and dimensions
- **Smart Interpolation**: Intelligent latent space interpolation between keyframes
- **Batch Conditioning**: Efficiently stacks multiple segments into unified conditioning data

## Parameters

### Required Inputs

#### `clip` (CLIP)
The CLIP model used for text encoding. This generates text embeddings that condition the video generation process.

#### `positive_prompt` (STRING, multiline)
Multi-line text prompts with optional keyframe indexing. Use the format `[N] prompt text` to assign specific prompts to keyframe segments.

**Example:**
```
[0] A peaceful mountain meadow at sunrise, soft golden light
[1] The same meadow at midday, bright blue sky, vibrant colors
[2] Evening scene with purple sunset, dramatic lighting
```

**Default:** `[0] A beautiful meadow\n[1] A misty forest\n[2] A glowing futuristic city`

#### `negative_prompt` (STRING)
Negative conditioning text applied to all segments to specify what should be avoided in the generation.

**Default:** `low quality, blurry, bad lighting`

#### `vae` (VAE)
The Variational Autoencoder model used for encoding images to latent space and determining spatial compression ratios.

#### `width` (INT)
Target video width in pixels.
- **Range:** 16 - 8192 pixels
- **Step:** 16 pixels
- **Default:** 496

#### `height` (INT)
Target video height in pixels.
- **Range:** 16 - 8192 pixels  
- **Step:** 16 pixels
- **Default:** 496

#### `fps` (INT)
Frames per second for the generated video.
- **Range:** 1 - 120 FPS
- **Default:** 16

#### `seconds` (INT)
Duration of each video segment in seconds.
- **Range:** 1 - 60 seconds
- **Default:** 1

### Optional Inputs

#### `keyframes` (IMAGE)
Input keyframe images that define the start and end points of video segments. The node requires at least 2 keyframes to create video segments. Each consecutive pair of keyframes creates one video segment.

**Requirements:**
- Minimum 2 keyframes required
- Images are automatically resized to match `width` and `height` parameters
- Supports batch input with multiple keyframes

#### `clip_vision_outputs` (CLIP_VISION_OUTPUT)
Optional CLIP vision embeddings for enhanced visual conditioning. When provided, these are merged with text conditioning to create more precise video generation guidance.

## Outputs

### `positive` (CONDITIONING)
Positive conditioning data containing:
- Stacked text embeddings from CLIP
- Pooled text outputs
- Concatenated latent images from keyframes
- Conditioning masks
- Optional CLIP vision embeddings

### `negative` (CONDITIONING) 
Negative conditioning data with the same structure as positive conditioning, using the negative prompt for all segments.

### `latent` (LATENT)
Prepared latent tensor ready for video generation, with proper batch and temporal dimensions.

## How It Works

### 1. Prompt Processing
The node parses the positive prompt for indexed entries using the pattern `[N] prompt text`. Each index corresponds to a keyframe segment:
- `[0]` applies to the segment from keyframe 0 to keyframe 1
- `[1]` applies to the segment from keyframe 1 to keyframe 2
- And so on...

If no indexed prompts are found, the first line is used for all segments.

### 2. Keyframe Segmentation
With N keyframes, the node creates N-1 video segments. Each segment represents the transition from one keyframe to the next.

### 3. Frame Processing
For each segment:
- Start and end keyframes are extracted and resized to target dimensions
- A video tensor is created with the specified length (fps × seconds + 1)
- Start frame is placed at the beginning, end frame at the end
- Intermediate frames are filled with neutral values for interpolation

### 4. Conditioning Generation  
Each segment gets:
- CLIP text encoding using the segment-specific prompt
- VAE latent encoding of the frame sequence
- Conditioning masks for proper temporal blending
- Optional CLIP vision processing if provided

### 5. Batch Stacking
All segment conditioning data is stacked into unified tensors that can be processed efficiently by video generation models.

## Usage Examples

### Basic Two-Keyframe Video
```python
# With 2 keyframes, creates 1 video segment
positive_prompt = "[0] A cat sitting in a garden, peaceful morning light"
# Segment 0: keyframe 0 → keyframe 1
```

### Multi-Segment Video
```python  
# With 4 keyframes, creates 3 video segments
positive_prompt = """[0] A butterfly on a flower, macro photography
[1] The butterfly taking flight, motion blur, dynamic
[2] Wide shot of garden with butterfly in distance, aerial view"""
# Segment 0: keyframe 0 → keyframe 1 (macro butterfly)
# Segment 1: keyframe 1 → keyframe 2 (taking flight)  
# Segment 2: keyframe 2 → keyframe 3 (wide garden view)
```

### Fallback Prompts
```python
# If keyframes don't have corresponding indexed prompts
positive_prompt = """[0] Default scene description
A forest path in different lighting conditions"""
# [0] used for segment 0, default text used for other segments
```

## Technical Notes

### Video Length Calculation
The actual video length is calculated as:
```python
length = (fps * seconds) + 1  # +1 for full second coverage
length = ((length - 1) // 4) * 4 + 1  # Round to multiple of 4 for compatibility
```

### Memory Considerations
- Video length directly impacts memory usage
- Higher resolutions require more VRAM
- More keyframes create more segments to process simultaneously
- Consider reducing parameters if encountering memory errors

### Compatibility
- Designed for video generation models that accept conditioning data
- Latent dimensions are automatically calculated based on VAE spatial compression
- Output tensors are formatted for standard ComfyUI video workflows

## Error Handling

### Common Issues

**"At least 2 keyframes are required"**
- Ensure your keyframes input contains at least 2 images
- Check that keyframes are properly connected in the workflow

**Memory errors**
- Reduce video resolution (`width`/`height`)
- Decrease `seconds` parameter  
- Use fewer keyframes
- Lower `fps` setting

**Conditioning stacking warnings**
- These are usually non-critical and indicate fallback behavior
- The node will use the first available conditioning as backup
- Results should still be functional

## Integration

This node is typically used in video generation workflows:

1. **Load Images From Directory** → keyframes input
2. **CLIP Text Encoder** → clip input  
3. **VAE Loader** → vae input
4. **WanKeyframeToVideo** → conditioning outputs
5. **Video Generation Model** (receives conditioning and latent outputs)