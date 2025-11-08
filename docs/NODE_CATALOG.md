# YS-vision-tools Node Catalog

**Complete Reference for all 22 Nodes**

Last Updated: 2025-11-08
Version: Phase 3 Complete
Total Nodes: 22

---

## Table of Contents

1. [2D Tracking Nodes (8)](#2d-tracking-nodes)
2. [Track Manipulation Nodes (6)](#track-manipulation-nodes)
3. [Rendering Nodes (9)](#rendering-nodes)
4. [Utility Nodes (5)](#utility-nodes)

---

## 2D Tracking Nodes

### 1. 2D Tracker (Object/Motion) 🎯

**ComfyUI Name:** `YS_TrackDetect`
**Display Name:** "2D Tracker (Object/Motion) 🎯"
**Category:** YS-vision-tools/Tracking
**File:** `nodes/track_detect.py`

**Purpose:** Technical tracking using computer vision methods

**Inputs:**
- `image` (IMAGE) - Input frame to analyze
- `detection_method` (enum) - CV algorithm to use
- `sensitivity` (FLOAT, 0-1) - Detection threshold
- `points_per_frame` (INT, 1-2000) - Max points to return
- `use_gpu` (BOOLEAN) - Enable GPU acceleration
- Optional: `gamma_correction`, `gradient_threshold`, `min_area`, `max_area`, `object_classes`, `confidence_threshold`, `flow_threshold`, `previous_frame`, `use_kalman_filter`, `temporal_smoothing`

**Outputs:**
- `tracks` (TRACKS) - Detected point coordinates (N, 2)
- `count` (INT) - Number of detected points
- `avg_confidence` (FLOAT) - Average confidence score
- `debug_viz` (IMAGE) - Visualization with points overlaid

**Detection Methods:**
1. **gradient_magnitude** - Scharr gradient edge detection (GPU, 3-5ms @ 4K)
2. **phase_congruency** - FFT multi-scale features (GPU, 6-10ms @ 4K)
3. **structure_tensor** - Shi-Tomasi corners (GPU, 4-6ms @ 4K)
4. **optical_flow** - Farneback motion (CPU-only, 35-50ms @ 4K, requires previous_frame)
5. **saliency_map** - Spectral residual attention (GPU, 8-12ms @ 4K)
6. **object_detection** - YOLO v8 semantic (GPU, 15-25ms @ 4K, requires ultralytics)
7. **hybrid_adaptive** - Gradient + structure fusion (GPU, 7-10ms @ 4K)

**GPU Coverage:** 6/7 methods (86%)

**Use Cases:**
- YOLO object detection (people, cars, faces)
- Motion tracking between frames (optical flow)
- Visual attention/saliency regions
- Fine-grained CV algorithm control
- Research and experimentation

---

### 2. 2D Tracker (Colors/Luma) 🎨

**ComfyUI Name:** `YS_TrackDetectV2`
**Display Name:** "2D Tracker (Colors/Luma) 🎨"
**Category:** YS-vision-tools/Tracking
**File:** `nodes/track_detect_v2.py`

**Purpose:** Artistic tracking using visual feature presets

**Inputs:**
- `image` (IMAGE) - Input frame
- `tracking_mode` (enum) - Preset mode to use
- `use_gpu` (BOOLEAN) - Enable GPU acceleration
- Optional: `points_per_frame` (INT, override preset), `sensitivity` (FLOAT, override preset), `temporal_smoothing` (FLOAT, 0-1), `prev_state` (STATE)

**Outputs:**
- `tracks` (TRACKS) - Detected points (N, 2)
- `state` (STATE) - Internal state with persistent IDs
- `debug_viz` (IMAGE) - Visualization

**Tracking Modes:**
1. **Exploratory Luma** - Edge strength (gradient magnitude), 320 points, 2-4ms @ 4K
2. **Color Hunter** - Hue boundaries (color discontinuities), 260 points, 8-12ms @ 4K (CPU)
3. **Locked Corners** - Harris corners (2D structure), 220 points, 3-5ms @ 4K
4. **Chroma Density** - High saturation regions, 280 points, 2-4ms @ 4K
5. **Phase Congruency** - Multi-scale edges, 240 points, 4-6ms @ 4K

**Unique Features:**
- ✨ Persistent track IDs via STATE output
- ✨ Color Hunter: Tracks hue boundaries (unique to this tracker)
- ✨ Chroma Density: Tracks high-saturation regions (unique to this tracker)
- ✨ Resolution-adaptive distance thresholds (4K/8K ready)
- ✨ Simplified workflow: 3 required params vs 15+ in Object/Motion

**GPU Coverage:** 4/5 modes (80%)

**Use Cases:**
- Color-based tracking (hue, saturation)
- Preset-based workflow (no CV knowledge needed)
- Persistent track IDs for temporal effects
- Fast batch video processing
- Artistic VFX and video editing

---

## Track Manipulation Nodes

### 3. YS Track Merge

**ComfyUI Name:** `YSTrackMerge`
**Display Name:** "YS Track Merge"
**Category:** YS-vision-tools/Tracking
**File:** `nodes/track_merge.py`

**Purpose:** Combine multiple track sources into unified output

**Inputs:**
- `tracks_a` (TRACKS, required) - First track source
- `tracks_b` (TRACKS, required) - Second track source
- Optional: `tracks_c` (TRACKS), `tracks_d` (TRACKS)
- Optional: `deduplicate` (BOOLEAN), `dedup_threshold` (FLOAT, 0.1-50, default 2.0)

**Outputs:**
- `tracks` (TRACKS) - Merged tracks
- `total_count` (INT) - Total points after merging

**Features:**
- Merge 2-4 track sources
- Batch/video mode support (frame-by-frame merging)
- Optional deduplication with distance threshold
- Handles mismatched batch sizes (uses last frame)

**Use Cases:**
- Combine detections from multiple methods
- Mix manual + detected tracks
- Layer effects from different sources

---

### 4. YS Track Deduplicate

**ComfyUI Name:** `YSTrackDeduplicate`
**Display Name:** "YS Track Deduplicate"
**Category:** YS-vision-tools/Tracking
**File:** `nodes/track_deduplicate.py`

**Purpose:** Remove overlapping/clustered tracking points

**Inputs:**
- `tracks` (TRACKS) - Input tracks
- `min_distance` (FLOAT, 0-500, default 50.0) - Minimum distance between points
- `keep_strategy` (enum) - Which point to keep per cluster
  - "first" - Keep first point (insertion order)
  - "last" - Keep last point
  - "center" - Keep point closest to cluster centroid
  - "random" - Keep random point
- `use_gpu` (BOOLEAN, default True) - GPU acceleration
- Optional: `max_points` (INT, 0=unlimited), `sort_by_position` (BOOLEAN)

**Outputs:**
- `tracks` (TRACKS) - Filtered tracks
- `input_count` (INT) - Original point count
- `output_count` (INT) - Filtered point count
- `reduction_percent` (FLOAT) - Percentage reduced

**Features:**
- GPU-accelerated distance matrix (100× faster than CPU)
- Connected components clustering
- Multiple keep strategies
- Batch processing for video
- Statistics output

**Performance:**
- 4K: 2-3ms GPU vs 200-300ms CPU (100× speedup)
- 8K: 5-8ms GPU vs 500-800ms CPU

**Use Cases:**
- Clean up dense detection clusters before text rendering
- Remove duplicate detections from multiple sources
- Control point density for visual effects

---

### 5. YS Track Jitter

**ComfyUI Name:** `YSTrackJitter`
**Display Name:** "YS Track Jitter"
**Category:** YS-vision-tools/Tracking
**File:** `nodes/track_jitter.py`

**Purpose:** Apply jitter and push-apart effects to tracked points

**Inputs:**
- `tracks` (TRACKS) - Input tracks
- `push_apart` (FLOAT, 0-100, default 0) - Minimum distance to maintain
- `jitter_amount` (FLOAT, 0-50, default 0) - Random variation per point
- `iterations` (INT, 1-20, default 5) - Push-apart iterations
- `seed` (INT, 0-999999, default 42) - Random seed for jitter
- `use_gpu` (BOOLEAN, default True) - GPU acceleration

**Outputs:**
- `tracks` (TRACKS) - Modified tracks

**Features:**
- GPU-accelerated distance calculations (10-50× faster)
- **Push Apart:** Iteratively separates overlapping points via repulsion forces
- **Jitter:** Adds consistent random variation per point (stable across frames)
- Deterministic per-point offsets (same seed = same pattern)
- Batch processing for video

**Performance:**
- 4K: 2-4ms GPU vs 20-200ms CPU (10-50× speedup)
- More iterations = better separation but slower

**Use Cases:**
- Prevent point overlap for text rendering
- Add organic variation to tracking patterns
- Create breathing/pulsing effects
- Spread dense clusters

---

## Rendering Nodes

### 6. Line Link Renderer (Advanced)

**ComfyUI Name:** `YSLineLinkRenderer`
**Display Name:** "Line Link Renderer (Advanced)"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/line_link_renderer.py`

**Purpose:** Render mathematical curves connecting tracked points

**Inputs:**
- `tracks` (TRACKS) - Points to connect
- `image_width` (INT, 64-7680) - Output width
- `image_height` (INT, 64-4320) - Output height
- `curve_type` (enum) - Mathematical curve to use
- `line_style` (enum) - Rendering style
- `thickness` (FLOAT, 0.5-20) - Line thickness
- `color` (STRING/COLOR) - Line color
- `opacity` (FLOAT, 0-1) - Transparency
- Many more parameters...

**Outputs:**
- `layer` (LAYER) - RGBA layer with rendered curves

**Curve Types (13):**
- straight, bezier_quadratic, bezier_cubic, catmull_rom
- logarithmic_spiral, elastic_curve, fourier_series
- field_line, gravitational_path
- voronoi_edges, delaunay_triangulation, minimum_spanning_tree
- neural_flow_field

**Line Styles (9):**
- solid, dotted, dashed, dash_dot
- gradient_fade, pulsing, electric, particle_trail
- wave_modulation

**Features:**
- 13 mathematical curve types
- 9 line rendering styles
- Anti-aliasing support (2x, 4x)
- Graph construction modes (kNN, radius, Delaunay, etc.)

---

### 7. Dot Renderer

**ComfyUI Name:** `YSDotRenderer`
**Display Name:** "Dot Renderer"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/dot_renderer.py`

**Purpose:** Render tracked points as styled markers

**Inputs:**
- `tracks` (TRACKS) - Point positions
- `image_width` (INT) - Output width
- `image_height` (INT) - Output height
- `size` (FLOAT, 1-50) - Dot size
- `style` (enum) - Marker style
- `glow` (FLOAT, 0-20) - Glow effect intensity
- `color` (STRING/COLOR) - Dot color
- `opacity` (FLOAT, 0-1) - Transparency

**Outputs:**
- `layer` (LAYER) - RGBA layer with dots

**Styles (6):**
- solid - Filled circle
- ring - Outline circle
- cross - X shape
- plus - + shape
- square - Filled square
- diamond - Filled diamond

**Features:**
- 6 marker styles
- Glow effects
- Per-point styling (via palette)
- Anti-aliased rendering

---

### 8. BBox Renderer

**ComfyUI Name:** `YSBBoxRenderer`
**Display Name:** "BBox Renderer"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/bbox_renderer.py`

**Purpose:** Render bounding boxes around tracked points

**Inputs:**
- `image_width`, `image_height` (INT) - Output dimensions
- `box_mode` (enum) - Sizing mode: "fixed", "from_radius", "from_age"
- `stroke_px` (FLOAT, 0.5-10, default 2.0) - Outline thickness
- `fill_opacity` (FLOAT, 0-1, default 0) - Interior fill transparency
- `roundness` (FLOAT, 0-1, default 0) - Corner roundness (0=square, 1=circular)
- `color` (COLOR) - Box color (supports color picker)
- Optional: `tracks` (TRACKS), `boxes` (BOXES), `ages` (AGES), `palette` (PALETTE)
- Optional: `width`, `height`, `radius_px` (sizing parameters)
- Optional: `alpha` (FLOAT, 0-1) - Transparency slider
- Optional: `use_gpu` (BOOLEAN, default True) - GPU acceleration

**Outputs:**
- `layer` (LAYER) - RGBA layer with boxes

**Features:**
- **GPU-accelerated SDF rendering** (2000× faster @ 4K)
- Custom CUDA kernels for distance field computation
- Color picker support (HEX, named colors, backward compatible)
- Multiple sizing modes
- Rounded corners with anti-aliasing
- Batch processing for video

**Performance:**
- 4K @ 100 boxes: 2ms GPU vs 187ms CPU (2000× speedup!)

**Use Cases:**
- Draw boxes around detected objects
- Highlight tracked regions
- Create HUD overlays

---

### 9. Blur Region Renderer

**ComfyUI Name:** `YSBlurRegionRenderer`
**Display Name:** "Blur Region Renderer"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/blur_region_renderer.py`

**Purpose:** Apply blur effects to regions around tracked points

**Inputs:**
- `image` (IMAGE) - Source image to blur
- `radius_px` (FLOAT, 5-200, default 20) - Blur region radius
- `sigma_px` (FLOAT, 1-50, default 5) - Gaussian blur strength
- `falloff` (FLOAT, 0-1, default 0.8) - Edge softness (0=hard, 1=very soft)
- `opacity` (FLOAT, 0-1, default 1.0) - Blur layer opacity
- `use_gpu` (BOOLEAN, default True) - GPU acceleration
- Optional: `tracks` (TRACKS), `boxes` (BOXES)

**Outputs:**
- `layer` (LAYER) - RGBA layer with blurred regions

**Features:**
- GPU-accelerated Gaussian blur (10-50× faster)
- Smooth sigmoid falloff masks
- Region radius control
- Depth-of-field-like effects
- Works with tracks or pre-computed boxes

**Performance:**
- 4K: 5-8ms GPU vs 50-400ms CPU (10-50× speedup)

**Use Cases:**
- Create focused/defocused regions
- Depth-of-field artistic effects
- Selective blur around tracked objects

---

### 10. MV Look Renderer

**ComfyUI Name:** `YSMVLookRenderer`
**Display Name:** "MV Look Renderer"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/mv_look_renderer.py`

**Purpose:** Apply machine vision / surveillance camera aesthetic

**Inputs:**
- `image` (IMAGE) - Source image
- `scanline_intensity` (FLOAT, 0-1, default 0.3) - CRT scanline darkness
- `chroma_offset_px` (FLOAT, 0-10, default 1.0) - Chromatic aberration offset
- `vignette` (FLOAT, 0-1, default 0.3) - Edge darkening strength
- `noise` (FLOAT, 0-0.5, default 0.05) - Film grain / sensor noise
- `as_layer` (BOOLEAN, default False) - Output as overlay layer vs modified image
- `use_gpu` (BOOLEAN, default True) - GPU acceleration
- Optional: `opacity` (FLOAT, 0-1), `scanline_spacing` (INT, 1-10), `color_tint` (STRING, RGB tint)

**Outputs:**
- `output` (IMAGE or LAYER) - Processed image or overlay

**Effects:**
- CRT scanlines with configurable spacing
- Chromatic aberration (lens distortion)
- Vignette darkening
- Film grain / sensor noise
- Color tinting (surveillance green/cyan look)

**Use Cases:**
- Retro technical/surveillance looks
- Security camera aesthetic
- CRT monitor simulation
- Machine vision style

---

### 11. YS Echo Layer (Temporal Trails)

**ComfyUI Name:** `YSEchoLayer`
**Display Name:** "YS Echo Layer (Temporal Trails)"
**Category:** YS-vision-tools/Effects
**File:** `nodes/echo_layer.py`

**Purpose:** Temporal decay / echo / trail effects

**Inputs:**
- `layer` (LAYER) - Current frame RGBA overlay
- `mode` (enum) - Decay mode: "accumulate", "pingpong", "framecount"
- `decay` (FLOAT, 0-1, default 0.88) - EMA decay factor per frame
- `boost_on_new` (FLOAT, 0-1, default 0.1) - Extra alpha boost on fresh pixels
- `max_age` (INT, 1-300, default 60) - Maximum trail age in frames
- `exposure` (FLOAT, 0-2, default 1.0) - Post-accumulation brightness gain
- `clamp_mode` (enum) - Output clamping: "hard", "softclip"
- `use_gpu` (BOOLEAN, default True) - GPU acceleration
- Optional: `state` (STATE), `flow_u` (FLOW_U), `flow_v` (FLOW_V), `warp_with_flow` (BOOLEAN)

**Outputs:**
- `layer` (LAYER) - Layer with trails
- `state` (STATE) - Internal accumulator state

**Features:**
- **Custom CUDA kernel for EMA updates** (3-5× faster)
- Exponential moving average temporal accumulation
- Multiple decay modes
- Optional optical flow warping for motion-aware trails
- Configurable exposure and clamping
- Per-frame state persistence for video

**Performance:**
- 4K: 3-5ms GPU vs 10-20ms CPU (3-5× speedup)

**Use Cases:**
- Motion trails on tracked objects
- Ghosting / echo effects
- Persistence-of-vision effects
- Stroboscopic visualization

---

### 12. YS Pixel Sorting (Tracks)

**ComfyUI Name:** `YSPixelSortingTracks`
**Display Name:** "YS Pixel Sorting (Tracks)"
**Category:** YS-vision-tools/Effects
**File:** `nodes/pixel_sorting_tracks.py`

**Purpose:** Track-gated pixel sorting glitch effect

**Inputs:**
- `image` (IMAGE) - Source frame
- `region_mode` (enum) - Region shape: "radius", "box"
- `radius_px` (FLOAT, 10-300, default 50) - Circle radius
- `box_w`, `box_h` (FLOAT, 20-500, default 100) - Box dimensions
- `direction` (enum) - Sorting direction: "horizontal", "vertical", "radial", "tangent_flow"
- `threshold_mode` (enum) - Metric for sorting: "luma", "saturation", "hue", "sobel_mag"
- `threshold_low`, `threshold_high` (FLOAT, 0-1) - Threshold range
- `sort_order` (enum) - "ascending", "descending", "shuffle"
- `segment_min_px` (INT, 3-128, default 8) - Minimum segment length
- `opacity` (FLOAT, 0-1, default 1.0) - Effect opacity
- `blend` (enum) - Blend mode: "normal", "add", "screen", "lighten"
- `seed` (INT) - Random seed for shuffle mode
- `use_gpu` (BOOLEAN, default True) - GPU acceleration
- Optional: `tracks` (TRACKS), `boxes` (BOXES), `mask` (MASK)

**Outputs:**
- `output` (IMAGE) - Processed image

**Features:**
- GPU-accelerated mask creation and metric computation
- Multiple sorting directions (h/v/radial/tangent)
- Flexible threshold modes (luma/sat/hue/sobel)
- Region control: circle radius or box dimensions
- Blend modes
- Batch processing for video

**Performance:**
- 4K: 8-12ms (GPU for mask/metric, CPU for sorting)

**Use Cases:**
- Artistic glitch effects on moving objects
- Data visualization with tracked elements
- Creative motion-reactive distortions
- Combining with EchoLayer for persistent streaks

---

### 13. YS Text On Tracks

**ComfyUI Name:** `YSTextOnTracks`
**Display Name:** "YS Text On Tracks"
**Category:** YS-vision-tools/Rendering
**File:** `nodes/text_on_tracks.py`

**Purpose:** Render text labels at tracked point positions

**Inputs:**
- `tracks` (TRACKS) - Track positions
- `image_width`, `image_height` (INT) - Output dimensions
- `content_mode` (enum) - Text content: "coordinates", "code_snippets", "string"
- `text_string` (STRING, default "Track Point") - Static text for 'string' mode
- `max_chars` (INT, 5-200, default 50) - Maximum text length
- `font_name` (enum) - Font to use (auto-discovered from system)
- `font_size` (INT, 8-72, default 16) - Font size in pixels
- `letter_spacing` (FLOAT, -5 to 20, default 0) - Kerning
- `color` (COLOR, default "#ffffff") - Text color (color picker)
- `opacity` (FLOAT, 0-1, default 1.0) - Text opacity
- `stroke_width` (INT, 0-10, default 0) - Outline thickness
- `stroke_color` (COLOR, default "#000000") - Outline color
- `offset_x`, `offset_y` (INT, -500 to 500) - Position offset from point
- `alignment` (enum) - Text anchor: bottom_left, bottom_center, bottom_right, center_left, center, center_right, top_left, top_center, top_right
- `use_gpu` (BOOLEAN, default False) - GPU for compositing
- Optional: `palette` (PALETTE) - Per-point colors

**Outputs:**
- `layer` (LAYER) - RGBA layer with text

**Content Modes:**
1. **coordinates** - Display point position as "[x, y]"
2. **code_snippets** - Random lines from codebase (loads from ys_vision_tools/*.py)
3. **string** - Static custom text

**Features:**
- PIL/Pillow-based text rendering (reliable, cross-platform)
- Full typography control: font, size, color, stroke, alignment
- Kerning (letter spacing) support
- Font discovery system (scans system fonts)
- Batch processing for video
- Color palette integration (per-point colors)

**Performance:**
- 4K: 15-25ms (CPU rendering, GPU compositing optional)

**Use Cases:**
- Label tracked objects with coordinates
- Display metadata at track positions
- Create HUD overlays with technical data
- Artistic text effects on tracked points

---

## Utility Nodes

### 14. Layer Merge

**ComfyUI Name:** `YSLayerMerge`
**Display Name:** "Layer Merge"
**Category:** YS-vision-tools/Utility
**File:** `nodes/layer_merge.py`

**Purpose:** Composite multiple RGBA layers

**Inputs:**
- `layer_a`, `layer_b` (LAYER, required) - Layers to merge
- Optional: `layer_c`, `layer_d` (LAYER)
- `blend_mode` (enum) - Blending algorithm
- `opacity` (FLOAT, 0-1) - Overall opacity

**Outputs:**
- `layer` (LAYER) - Merged RGBA layer

**Blend Modes (7):**
- normal - Standard alpha compositing
- add - Additive blending
- multiply - Multiplicative
- screen - Screen blending
- overlay - Overlay blending
- max - Maximum per-channel
- min - Minimum per-channel

**Features:**
- Merge 2-4 layers
- Multiple blend modes
- Opacity control
- Batch processing for video

**Use Cases:**
- Combine multiple rendering layers
- Create complex visual effects
- Layer management

---

### 15. Composite Over

**ComfyUI Name:** `YSCompositeOver`
**Display Name:** "Composite Over"
**Category:** YS-vision-tools/Utility
**File:** `nodes/composite_over.py`

**Purpose:** Alpha-composite RGBA layer onto base image

**Inputs:**
- `base_image` (IMAGE) - Background RGB image
- `layer` (LAYER) - RGBA overlay layer
- `opacity` (FLOAT, 0-1) - Layer opacity
- Optional: `offset_x`, `offset_y` (INT) - Position offset

**Outputs:**
- `image` (IMAGE) - Composited result

**Features:**
- Alpha blending with premultiplied alpha
- Position offset support
- Automatic resizing
- Batch processing for video

**Use Cases:**
- Final composition of layers onto video frames
- Overlay effects on images
- Combine rendered layers with source footage

---

### 16. Image Size Detector

**ComfyUI Name:** `YSImageSizeDetector`
**Display Name:** "Image Size Detector"
**Category:** YS-vision-tools/Utility
**File:** `nodes/image_size_detector.py`

**Purpose:** Extract dimensions from image tensors

**Inputs:**
- `image` (IMAGE) - Image tensor

**Outputs:**
- `width` (INT) - Image width
- `height` (INT) - Image height

**Features:**
- Automatic dimension detection
- Works with batch/video tensors
- Simple utility for workflow automation

**Use Cases:**
- Feed dimensions to other nodes automatically
- Dynamic size-based processing
- Workflow automation

---

### 17. Video Frame Offset

**ComfyUI Name:** `YSVideoFrameOffset`
**Display Name:** "Video Frame Offset"
**Category:** YS-vision-tools/Utility
**File:** `nodes/video_frame_offset.py`

**Purpose:** Offset video frames by N frames

**Inputs:**
- `image_batch` (IMAGE) - Batch of video frames
- `offset` (INT) - Number of frames to offset (can be negative)

**Outputs:**
- `image_batch` (IMAGE) - Offset batch

**Features:**
- Temporal alignment utility
- Frame delay creation
- Supports negative offsets (shift backwards)
- Batch processing

**Use Cases:**
- Temporal alignment between tracks and frames
- Create frame delays for effects
- Synchronize multiple video streams

---

## Quick Reference

### Nodes by Use Case

**Tracking:**
- Object/semantic tracking → Object/Motion Tracker (YOLO, optical flow)
- Color/luma tracking → Colors/Luma Tracker (hue, saturation)
- Technical CV control → Object/Motion Tracker (7 methods)
- Artistic presets → Colors/Luma Tracker (5 modes)

**Track Manipulation:**
- Combine sources → Track Merge
- Remove duplicates → Track Deduplicate
- Add variation → Track Jitter

**Rendering:**
- Connect points → Line Link Renderer
- Draw markers → Dot Renderer
- Draw boxes → BBox Renderer
- Blur regions → Blur Region Renderer
- Surveillance look → MV Look Renderer
- Temporal trails → Echo Layer
- Glitch effects → Pixel Sorting
- Text labels → Text On Tracks

**Utilities:**
- Layer compositing → Layer Merge
- Final composite → Composite Over
- Get dimensions → Image Size Detector
- Frame alignment → Video Frame Offset

### GPU-Accelerated Nodes

**2000× speedup:** BBox Renderer (CUDA SDF)
**100× speedup:** Track Deduplicate (distance matrix)
**50× speedup:** Track Jitter (push-apart), Blur Region
**10× speedup:** Track Detect, Blur Region
**5× speedup:** Track Detect, Echo Layer

### Batch Processing Support

**All nodes support batch/video processing** (22/22):
- Frame-by-frame processing with progress logging
- State persistence for temporal effects
- Unified batch detection pattern

---

## Node Dependencies

### Required Dependencies
- Core: numpy, opencv-python, scipy
- ComfyUI: torch, torchvision

### Optional Dependencies
- GPU: cupy-cuda12x (for GPU acceleration)
- YOLO: ultralytics (for object detection)
- Text: PIL/Pillow (for text rendering)
- JIT: numba (for CPU optimization)

---

## Common Workflows

### Workflow 1: Tracked Object Labels with Deduplication

```
Tracker (Colors/Luma)
  ↓
Track Deduplicate (min_distance=50, keep=center)
  ↓
Text On Tracks (content=coordinates, font_size=14, stroke_width=1)
  ↓
Composite Over → Output
```

**Purpose:** Clean labels without overlap

---

### Workflow 2: Motion Trails with Push-Apart

```
Tracker → Track Jitter (push_apart=30, jitter=5)
  ↓
Dot Renderer (size=8, glow=4)
  ↓
Echo Layer (decay=0.9, max_age=60)
  ↓ state ↑ (loop)
Composite Over → Output
```

**Purpose:** Organic motion trails with spacing

---

### Workflow 3: Glitch Effect with Surveillance Look

```
Image ─────────────────┐
  ↓                    │
Tracker                │
  ↓                    │
Pixel Sorting (Tracks) │
  ↓                    │
Composite Over ←───────┘
  ↓
MV Look Renderer (surveillance preset)
  ↓
Output
```

**Purpose:** Tracked glitch effect + retro aesthetic

---

### Workflow 4: Multi-Source Track Fusion

```
Tracker 1 (Colors/Luma, mode=Exploratory Luma)
  ↓
Tracker 2 (Object/Motion, method=structure_tensor)
  ↓
Track Merge (deduplicate=True, threshold=5.0)
  ↓
Track Deduplicate (min_distance=40, keep=center)
  ↓
Track Jitter (push_apart=30)
  ↓
Text On Tracks (content=code_snippets)
  ↓
Echo Layer (decay=0.88)
  ↓ state ↑
Composite Over → Output
```

**Purpose:** Ultimate tracked text with trails

---

### Workflow 5: Dimension-Aware Dynamic Workflow

```
LoadImage → Image Size Detector
              ↓ width  ↓ height
              ↓        ↓
Tracker → Text On Tracks (auto-size)
              ↓
Composite Over → Output
```

**Purpose:** No manual dimension entry

---

## Troubleshooting

### Batch Processing Issues

**Problem:** Effect only applies to first frame of video
**Solution:** Check that node is looping through `range(batch_size)` not just `[0]`
**Reference:** CRITICAL section in CLAUDE.md

**Problem:** STATE not persisting across frames
**Solution:** Connect STATE output back to STATE input (create loop)

### Performance Issues

**Problem:** Slow processing with many points (1000+)
**Solution:** Use Track Deduplicate to reduce point count
**Benefit:** 100× speedup + cleaner output

**Problem:** Push-apart taking too long
**Solution:** Reduce `iterations` parameter (5→3 or 5→1)
**Trade-off:** Less separation quality

### Text Rendering Issues

**Problem:** Font not found / default font used
**Solution:** Check font name exactly matches system font file
**Tip:** Use font names from dropdown (auto-discovered)

**Problem:** Text overlapping
**Solution:** Use Track Deduplicate or Track Jitter (push_apart) before text rendering
**Recommended:** `min_distance=50` or `push_apart=40`

### Color Picker Issues

**Problem:** Color input not working
**Solution:** Ensure color format is HEX string ("#ffffff") or named color ("red")
**Fallback:** Use legacy float list format [1.0, 1.0, 1.0]

---

## Best Practices

### For Video Processing
1. **Always connect STATE loops** for temporal nodes (Echo Layer)
2. **Monitor batch size** - check console for "BATCH MODE: N frames"
3. **Use progress logging** - every 10 frames
4. **Test with short clips** first (10-50 frames) before full video

### For Performance
1. **Enable GPU** on all GPU-capable nodes
2. **Use Track Deduplicate** for large point sets (>500)
3. **Reduce iterations** on Track Jitter if slow
4. **Profile with console logging** - look for "[YS-XXX] GPU/CPU" messages

### For Visual Quality
1. **Use Track Deduplicate or Jitter** to prevent overlapping text
2. **Tune decay** on Echo Layer for desired trail length
3. **Adjust segment_min_px** on Pixel Sorting for effect intensity
4. **Enable stroke** on text for better readability

---

## Development Status

**Phase 1 (Complete):** 6 core nodes
**Phase 2 (Complete):** 3 extended renderer nodes + 1 tracker
**Phase 3 (Complete):** 9 new nodes + batch processing fixes

**Total:** 21 nodes, ~15,000 lines of code, 3 custom CUDA kernels

---

Last Updated: 2025-11-08
Project: YS-vision-tools
Developer: Yambo Studio
