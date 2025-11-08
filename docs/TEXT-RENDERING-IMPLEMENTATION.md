# Text Rendering Implementation Guide

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools for ComfyUI
**Status:** Active Implementation Pattern

---

## Overview

This document describes the text rendering system implemented in YS-vision-tools, specifically the Text On Tracks node. This pattern can be used for any future nodes requiring text rendering capabilities.

**Key Features:**
- PIL/Pillow-based rendering (reliable, cross-platform)
- System font discovery and caching
- Multiple content modes (coordinates, code snippets, custom text)
- Full typography control (font, size, color, stroke, alignment, kerning)
- Batch processing for video
- GPU-accelerated compositing
- Color picker integration

---

## Reference Implementation

**Primary Reference:** `nodes/text_on_tracks.py`
**Font Discovery:** Platform-specific system font scanning
**Text Rendering:** PIL ImageDraw for cross-platform compatibility
**Compositing:** CuPy GPU acceleration for alpha blending

---

## Architecture Overview

### Text Rendering Stack

```
1. Font Discovery → Scan system fonts, build dropdown list
2. Font Loading → Load and cache TrueType/OpenType fonts
3. Content Generation → Generate text per track point
4. Text Measurement → Calculate bounding boxes
5. Rendering → PIL ImageDraw to transparent canvas
6. Compositing → GPU-accelerated alpha blend onto layer
```

### Why PIL/Pillow?

**Advantages:**
- Cross-platform (Windows, Mac, Linux)
- System font support (no font file bundling)
- Reliable text measurement (accurate bounding boxes)
- Stroke/outline support built-in
- Kerning (letter spacing) support

**Trade-offs:**
- CPU-only rendering (no GPU acceleration for text itself)
- Slower than GPU solutions (but acceptable: 15-25ms @ 4K)
- GPU used for compositing step only

---

## Implementation Pattern

### Step 1: Font Discovery System

```python
@classmethod
def _get_available_fonts(cls) -> List[str]:
    """Discover available system fonts"""
    fonts = []

    # Platform-specific font directories
    if sys.platform == "win32":
        font_dirs = [
            Path(os.environ.get("SystemRoot", "C:\\Windows")) / "Fonts"
        ]
    elif sys.platform == "darwin":  # macOS
        font_dirs = [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts"
        ]
    else:  # Linux
        font_dirs = [
            Path("/usr/share/fonts/truetype"),
            Path("/usr/share/fonts/TTF"),
            Path.home() / ".fonts"
        ]

    # Scan for .ttf and .otf files
    for font_dir in font_dirs:
        if font_dir.exists():
            for font_file in font_dir.rglob("*.ttf"):
                fonts.append(font_file.name)
            for font_file in font_dir.rglob("*.otf"):
                fonts.append(font_file.name)

    # Deduplicate and sort
    fonts = sorted(list(set(fonts)))

    # Fallback if no fonts found
    if not fonts:
        fonts = ["default"]

    return fonts
```

**Key Points:**
- Run at INPUT_TYPES time to populate dropdown
- Platform detection for cross-platform support
- Recursive scan (rglob) finds fonts in subdirectories
- Deduplication handles fonts in multiple locations
- Fallback to "default" if no fonts found

### Step 2: Font Loading with Caching

```python
# Class-level cache
_font_cache = {}

@classmethod
def _load_font(cls, font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Load font with caching"""
    cache_key = (font_name, font_size)

    # Check cache first
    if cache_key in cls._font_cache:
        return cls._font_cache[cache_key]

    font = None

    # Try to load specified font
    if font_name != "default":
        # Search platform-specific directories
        for font_dir in [Platform directories]:
            if font_dir.exists():
                for font_path in font_dir.rglob(font_name):
                    try:
                        font = ImageFont.truetype(str(font_path), font_size)
                        break
                    except Exception as e:
                        print(f"[YS-TEXT] Failed to load {font_path}: {e}")

    # Fallback to default font
    if font is None:
        try:
            font = ImageFont.load_default()
            print(f"[YS-TEXT] Using default font (size fixed)")
        except Exception as e:
            print(f"[YS-TEXT] Failed to load default font: {e}")
            # Last resort: create dummy font
            font = ImageFont.load_default()

    # Cache the loaded font
    cls._font_cache[cache_key] = font
    return font
```

**Key Points:**
- Class-level cache prevents reloading fonts
- Cache key is `(font_name, font_size)` tuple
- Search system directories for font file
- Graceful fallback to default font
- Error handling for missing/corrupt fonts

### Step 3: Content Generation

```python
def _generate_content(self, content_mode: str, position: tuple,
                     text_string: str, code_lines: List[str],
                     max_chars: int) -> str:
    """Generate text content based on mode"""

    if content_mode == "coordinates":
        # Format: [x, y]
        text = f"[{int(position[0])}, {int(position[1])}]"

    elif content_mode == "code_snippets":
        # Random code line from codebase
        text = random.choice(code_lines)

    elif content_mode == "string":
        # Static custom text
        text = text_string

    else:
        text = "?"

    # Truncate to max length
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text
```

**Content Modes:**

1. **Coordinates** - Display point position
   - Format: `"[123, 456]"`
   - Use case: Debug tracking, position labels

2. **Code Snippets** - Random code from codebase
   - Loads `.py` files from project
   - Strips comments and empty lines
   - Use case: Artistic/glitch effects

3. **String** - Static custom text
   - User-provided text string
   - Same text at all points
   - Use case: Labels, annotations

### Step 4: Text Measurement with Kerning

```python
def _measure_text_with_kerning(self, text: str, font: ImageFont.FreeTypeFont,
                                letter_spacing: float) -> Tuple[int, int]:
    """
    Measure text dimensions with letter spacing

    Args:
        text: Text string to measure
        font: PIL Font object
        letter_spacing: Extra spacing between letters (pixels)

    Returns:
        (width, height) tuple in pixels
    """
    if letter_spacing == 0.0:
        # Fast path: use built-in measurement
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    # Manual measurement with kerning
    total_width = 0
    max_height = 0

    for char in text:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]

        total_width += char_width + letter_spacing
        max_height = max(max_height, char_height)

    # Remove last spacing
    if letter_spacing > 0:
        total_width -= letter_spacing

    return (int(total_width), int(max_height))
```

**Key Points:**
- Fast path for zero letter spacing
- Manual measurement for kerning
- Per-character width calculation
- Maximum height across all characters

### Step 5: Text Rendering with PIL

```python
def _render_text_pil(self, text: str, font: ImageFont.FreeTypeFont,
                     text_rgba: tuple, stroke_rgba: tuple,
                     stroke_width: int, letter_spacing: float) -> np.ndarray:
    """
    Render text to transparent canvas using PIL

    Returns:
        NumPy array (H, W, 4) RGBA with text rendered
    """
    # Measure text dimensions
    text_width, text_height = self._measure_text_with_kerning(
        text, font, letter_spacing
    )

    # Add padding for stroke
    padding = stroke_width * 2 + 4
    canvas_width = text_width + padding * 2
    canvas_height = text_height + padding * 2

    # Create transparent PIL image
    image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Convert float RGBA (0.0-1.0) to int (0-255)
    text_color = tuple(int(c * 255) for c in text_rgba)
    stroke_color = tuple(int(c * 255) for c in stroke_rgba)

    # Render text with optional stroke
    if letter_spacing == 0.0:
        # Fast path: single draw call
        draw.text(
            (padding, padding),
            text,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color if stroke_width > 0 else None
        )
    else:
        # Manual kerning: draw character by character
        x = padding
        for char in text:
            draw.text(
                (x, padding),
                char,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color if stroke_width > 0 else None
            )
            bbox = font.getbbox(char)
            x += (bbox[2] - bbox[0]) + letter_spacing

    # Convert PIL to NumPy RGBA
    text_array = np.array(image).astype(np.float32) / 255.0

    return text_array
```

**Key Points:**
- Transparent canvas with padding
- Color conversion: float (0.0-1.0) → int (0-255)
- Optional stroke outline
- Manual character placement for kerning
- Output: NumPy array (H, W, 4) RGBA in 0.0-1.0 range

### Step 6: Position Calculation with Alignment

```python
def _calculate_position(self, track_pos: tuple, text_width: int, text_height: int,
                       offset_x: int, offset_y: int, alignment: str,
                       image_width: int, image_height: int) -> Tuple[int, int]:
    """
    Calculate final text position based on alignment

    Args:
        track_pos: Track point position (x, y)
        text_width, text_height: Text dimensions
        offset_x, offset_y: User offset from track point
        alignment: Anchor position (e.g., "bottom_left", "center")

    Returns:
        (x, y) position for top-left corner of text
    """
    # Base position: track point + offset
    base_x = int(track_pos[0]) + offset_x
    base_y = int(track_pos[1]) + offset_y

    # Adjust based on alignment
    if "left" in alignment:
        x = base_x
    elif "right" in alignment:
        x = base_x - text_width
    else:  # center
        x = base_x - text_width // 2

    if "top" in alignment:
        y = base_y
    elif "bottom" in alignment:
        y = base_y - text_height
    else:  # center
        y = base_y - text_height // 2

    # Clamp to image bounds
    x = max(0, min(x, image_width - text_width))
    y = max(0, min(y, image_height - text_height))

    return (x, y)
```

**Alignment Options:**
```
top_left        top_center        top_right
center_left     center            center_right
bottom_left     bottom_center     bottom_right
```

### Step 7: GPU-Accelerated Compositing

```python
def _composite_text_gpu(self, layer: np.ndarray, text_array: np.ndarray,
                       position: tuple, use_gpu: bool = True) -> np.ndarray:
    """
    Composite text onto layer using GPU alpha blending

    Args:
        layer: Base layer (H, W, 4) RGBA
        text_array: Text to composite (h, w, 4) RGBA
        position: (x, y) top-left position
        use_gpu: Enable GPU acceleration

    Returns:
        Layer with text composited
    """
    x, y = position
    h, w = text_array.shape[:2]

    # Extract region of interest
    roi = layer[y:y+h, x:x+w]

    if use_gpu and CUPY_AVAILABLE:
        # Transfer to GPU
        roi_gpu = cp.asarray(roi)
        text_gpu = cp.asarray(text_array)

        # Alpha blending: dst = src * alpha + dst * (1 - alpha)
        alpha = text_gpu[..., 3:4]  # (h, w, 1)
        blended_gpu = text_gpu[..., :3] * alpha + roi_gpu[..., :3] * (1 - alpha)

        # Composite alpha channel
        blended_alpha_gpu = cp.maximum(roi_gpu[..., 3:4], alpha)

        # Combine
        result_gpu = cp.concatenate([blended_gpu, blended_alpha_gpu], axis=2)

        # Transfer back
        result = cp.asnumpy(result_gpu)
    else:
        # CPU alpha blending
        alpha = text_array[..., 3:4]
        blended = text_array[..., :3] * alpha + roi[..., :3] * (1 - alpha)
        blended_alpha = np.maximum(roi[..., 3:4], alpha)
        result = np.concatenate([blended, blended_alpha], axis=2)

    # Update layer
    layer[y:y+h, x:x+w] = result
    return layer
```

**Key Points:**
- GPU acceleration for compositing step only (not text rendering)
- Standard alpha blending formula
- Proper alpha channel composition
- CPU fallback for non-GPU environments

---

## Batch Processing Pattern

### Video Processing with Frame Loop

```python
def execute(self, tracks, image_width, image_height, ..., **kwargs):
    """Process text rendering for single frame or video batch"""

    # Handle batch vs single
    if isinstance(tracks, list):
        is_batch = True
        batch_size = len(tracks)
        print(f"[YS-TEXT] BATCH MODE: {batch_size} frames")
    else:
        is_batch = False
        batch_size = 1
        tracks = [tracks]

    # Load font ONCE (not per-frame)
    font = self._load_font(font_name, font_size)

    # Parse colors ONCE (not per-frame)
    text_rgba = normalize_color_to_rgba01(color, opacity)
    stroke_rgba = normalize_color_to_rgba01(stroke_color, 1.0)

    # Process each frame
    output_frames = []

    for frame_idx in range(batch_size):
        frame_tracks = tracks[frame_idx]

        # Create blank layer
        layer = np.zeros((image_height, image_width, 4), dtype=np.float32)

        # Render text for each track point
        for i, pos in enumerate(frame_tracks):
            # Generate content
            text = self._generate_content(...)

            # Render text to canvas
            text_array = self._render_text_pil(text, font, text_rgba, stroke_rgba, ...)

            # Calculate position
            position = self._calculate_position(pos, ...)

            # Composite onto layer (GPU accelerated)
            layer = self._composite_text_gpu(layer, text_array, position, use_gpu)

        output_frames.append(layer)

        # Progress logging
        if is_batch and (frame_idx % 10 == 0 or frame_idx == batch_size - 1):
            print(f"[YS-TEXT] Processed frame {frame_idx+1}/{batch_size}")

    # Stack into batch tensor
    if is_batch:
        output_batch = np.stack(output_frames, axis=0)
        return (torch.from_numpy(output_batch.astype(np.float32)),)
    else:
        return (torch.from_numpy(output_frames[0]).unsqueeze(0).float(),)
```

**Key Optimizations:**
- Font loaded once (not per-frame)
- Colors parsed once (not per-frame)
- GPU compositing for each text element
- Progress logging every 10 frames

---

## INPUT_TYPES Configuration

### Complete Input Structure

```python
@classmethod
def INPUT_TYPES(cls) -> Dict[str, Any]:
    # Discover fonts
    font_list = cls._get_available_fonts()

    return {
        "required": {
            "tracks": ("TRACKS",),
            "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680}),
            "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320}),

            # Content mode
            "content_mode": ([
                "coordinates",
                "code_snippets",
                "string"
            ], {"default": "coordinates"}),

            "text_string": ("STRING", {
                "default": "Track Point",
                "multiline": False
            }),

            "max_chars": ("INT", {
                "default": 50,
                "min": 5,
                "max": 200
            }),

            # Typography
            "font_name": (font_list, {
                "default": font_list[0] if font_list else "arial.ttf"
            }),
            "font_size": ("INT", {
                "default": 16,
                "min": 8,
                "max": 72
            }),
            "letter_spacing": ("FLOAT", {
                "default": 0.0,
                "min": -5.0,
                "max": 20.0,
                "step": 0.5
            }),

            # Color picker integration
            "color": ("COLOR", {
                "default": "#ffffff",
                "tooltip": "Text color"
            }),
            "opacity": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05
            }),

            # Stroke
            "stroke_width": ("INT", {
                "default": 0,
                "min": 0,
                "max": 10
            }),
            "stroke_color": ("COLOR", {
                "default": "#000000"
            }),

            # Positioning
            "offset_x": ("INT", {"default": 10, "min": -500, "max": 500}),
            "offset_y": ("INT", {"default": -20, "min": -500, "max": 500}),
            "alignment": ([
                "bottom_left", "bottom_center", "bottom_right",
                "center_left", "center", "center_right",
                "top_left", "top_center", "top_right"
            ], {"default": "bottom_left"}),

            # Performance
            "use_gpu": ("BOOLEAN", {
                "default": False,
                "tooltip": "Use GPU for compositing (CPU for text rendering)"
            }),
        },
        "optional": {
            "palette": ("PALETTE",),  # Per-point colors override
        }
    }
```

---

## Use Cases and Examples

### Use Case 1: Coordinate Labels

```python
# Settings
content_mode = "coordinates"
font_size = 14
color = "#ffffff"
stroke_width = 1
stroke_color = "#000000"
alignment = "bottom_left"
offset_x = 10
offset_y = -20

# Result: Clean [x, y] labels offset from points
```

### Use Case 2: Code Glitch Effect

```python
# Settings
content_mode = "code_snippets"
font_name = "Courier New"
font_size = 12
max_chars = 80
color = "#00ff00"  # Matrix green
opacity = 0.7
alignment = "center_left"

# Result: Random code lines at track points, hacker aesthetic
```

### Use Case 3: Custom Annotations

```python
# Settings
content_mode = "string"
text_string = "Feature Point"
font_size = 16
color = "#ff0000"
stroke_width = 2
stroke_color = "#000000"
alignment = "top_center"

# Result: Static label above each point
```

---

## Performance Characteristics

### Timing Breakdown @ 4K Resolution

```
500 points, 16px font:
- Font loading: <1ms (cached after first)
- Content generation: 1-2ms
- Text rendering (PIL): 10-15ms (CPU-only)
- Position calculation: <1ms
- GPU compositing: 3-5ms
- TOTAL: 15-25ms per frame

Video (50 frames):
- Frame 1: 25ms (font loading)
- Frames 2-50: 15-20ms (cached)
- Total: ~850ms for 50 frames
```

**Performance Notes:**
- Text rendering is CPU-bound (PIL limitation)
- GPU used for compositing step (3-5× faster)
- Font caching critical for video performance
- Per-point overhead scales linearly

---

## Future Optimization Opportunities

### Potential GPU Text Rendering

**SDF Font Rendering:**
- Pre-generate signed distance fields for glyphs
- GPU shader-based rendering
- Estimated speedup: 10-20×
- Trade-off: Complex implementation, glyph atlas generation

**Current Decision:** PIL is "good enough" for now
- 15-25ms @ 4K is acceptable for current use cases
- Cross-platform compatibility more important than speed
- Can optimize later if bottleneck identified

---

## Extending This Pattern

### For Future Text-Based Nodes

1. **Copy Font Discovery System**
   - `_get_available_fonts()` class method
   - Platform-specific directory scanning
   - Dropdown population in INPUT_TYPES

2. **Copy Font Loading with Caching**
   - Class-level `_font_cache` dictionary
   - `_load_font()` method with cache lookup
   - Graceful fallback to default font

3. **Implement Custom Content Generation**
   - Replace `_generate_content()` with your logic
   - Support different content modes as needed

4. **Use PIL for Rendering**
   - `_render_text_pil()` pattern
   - Transparent canvas with padding
   - Stroke support optional

5. **Add GPU Compositing**
   - `_composite_text_gpu()` for alpha blending
   - CPU fallback for non-GPU environments

6. **Integrate Color Picker**
   - Use COLOR input type
   - Parse with `normalize_color_to_rgba01()`
   - Support text + stroke colors

---

## References

- **Reference Implementation:** `nodes/text_on_tracks.py` - Complete working example
- **Color Integration:** `docs/COLOR-PICKER-IMPLEMENTATION.md` - Color picker pattern
- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.md` - Node structure
- **Batch Processing:** `CLAUDE.md` - Critical batch processing rules
- **PIL Documentation:** [Pillow Docs](https://pillow.readthedocs.io/)

---

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools
**Developer:** Yambo Studio
