# YS-vision-tools

Advanced ComfyUI custom nodes for GPU-accelerated vision overlays with experimental mathematical curves and smart detection methods.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![CUDA](https://img.shields.io/badge/CUDA-12.x-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Overview

A visual effects system for ComfyUI providing GPU-accelerated 2D tracking and mathematical curve rendering. Designed for RTX 5090 performance targets (4K @ 60fps).

**Core capabilities:**
- **2D Tracker (Object/Motion)**: Track using 7 CV methods (gradient, optical flow, phase congruency, structure tensor, saliency, YOLO, hybrid) - 86% GPU-accelerated
- **2D Tracker (Colors/Luma)**: Track using 5 visual presets (luma edges, color boundaries, corners, saturation, multi-scale) - 80% GPU-accelerated
- Render connections with 15+ curve types (Bézier, spirals, field lines, Fourier series, elastic curves)
- Apply 10+ line styles (solid, electric, pulsing, particle trails, wave modulation)
- Compose layers with blend modes and alpha compositing

![example workflow](images/example-workflow.png)

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yamb0x/comfyui-ys-vision-tools.git
cd comfyui-ys-vision-tools
pip install -r requirements.txt
```

**GPU acceleration (recommended):**
```bash
pip install cupy-cuda12x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics  # for YOLO detection
```

Restart ComfyUI after installation.

## Available Nodes

**Total: 21 nodes across 4 categories**

### 2D Tracking Nodes (8 nodes)

YS-vision-tools provides two complementary 2D tracking nodes plus 6 track manipulation nodes:

---

### 2D Tracker (Object/Motion) 🎯
**For technical tracking and motion analysis**

Tracks features using 7 computer vision methods with full GPU acceleration (6/7 methods).

**Detection Methods:**

| Method | Description | GPU | Speed @ 4K | Best For |
|--------|-------------|-----|------------|----------|
| **gradient_magnitude** | Scharr gradient edges with NMS | ✅ Full | 3-5ms | General edge tracking, fast |
| **phase_congruency** | FFT multi-scale feature detection | ✅ Full | 6-10ms | Lighting-invariant patterns |
| **structure_tensor** | Shi-Tomasi corner detection | ✅ Full | 4-6ms | Corners, keypoints |
| **optical_flow** | Farneback motion tracking | ⚠️ CPU-only | 35-50ms | Motion between frames (requires previous_frame) |
| **saliency_map** | Spectral residual visual attention | ✅ Full | 8-12ms | Perceptually salient regions |
| **object_detection** | YOLO v8 semantic detection | ✅ Full | 15-25ms | People, cars, faces (requires ultralytics) |
| **hybrid_adaptive** | Combines gradient + structure | ✅ Full | 7-10ms | Best accuracy, multi-method fusion |

**Inputs:** Image, detection method, sensitivity, max points, area filter, previous_frame (optical flow), use_gpu  
**Outputs:** Track coordinates, count, confidence, debug visualization  

**Parameters (15+ controls):**
- `detection_method`: Choose from 7 CV algorithms
- `sensitivity`: Detection threshold (0.0-1.0)
- `points_per_frame`: Max tracks to return
- `gamma_correction`: Preprocessing adjustment
- `gradient_threshold`, `flow_threshold`: Method-specific tuning
- `min_area`, `max_area`: Area filtering
- `object_classes`, `confidence_threshold`: YOLO options
- `use_kalman_filter`, `temporal_smoothing`: Temporal tracking
- `use_gpu`: Enable GPU acceleration (6/7 methods)

**When to use:**
- ✅ Need YOLO object detection (people, cars, specific classes)
- ✅ Need motion tracking with optical flow
- ✅ Want visual attention/saliency detection
- ✅ Need fine-grained control over CV algorithms
- ✅ Research, experimentation, technical workflows

**GPU Performance:** 86% coverage (6/7 methods fully GPU-accelerated)

---

### 2D Tracker (Colors/Luma) 🎨
**For artistic tracking and visual effects**

Tracks features using 5 preset modes with simplified controls and full GPU acceleration (4/5 modes).

**Tracking Modes:**

| Mode | Tracks | GPU | Speed @ 4K | Use Case |
|------|--------|-----|------------|----------|
| **Exploratory Luma** | Edge strength (gradients) | ✅ Full | 2-4ms | General edge tracking |
| **Color Hunter** | Hue boundaries (color changes) | ⚠️ CPU | 8-12ms | Color transitions, boundaries |
| **Locked Corners** | Harris corners (2D structure) | ✅ Full | 3-5ms | Sharp features, geometric |
| **Chroma Density** | High saturation regions | ✅ Full | 2-4ms | Colorful objects, vibrant areas |
| **Phase Congruency** | Multi-scale edges | ✅ Full | 4-6ms | Complex textures, fine details |

**Inputs:** Image, tracking_mode, use_gpu, points_per_frame (optional), sensitivity (optional), temporal_smoothing (optional), prev_state  
**Outputs:** Track coordinates, state (with persistent IDs), debug visualization

**Parameters (3 required + 3 optional):**
- `tracking_mode`: Select visual preset (5 modes)
- `use_gpu`: Enable GPU acceleration (4/5 modes)
- `points_per_frame`: Override preset count (0 = use preset)
- `sensitivity`: Override preset sensitivity (0 = use preset)
- `temporal_smoothing`: Smooth tracks across frames (0.0-1.0)
- `prev_state`: Internal state for video (loop with STATE output)

**Unique Features:**
- ✨ **Color Hunter**: Tracks hue boundaries (no equivalent in Object/Motion tracker)
- ✨ **Chroma Density**: Tracks high-saturation regions (no equivalent in Object/Motion tracker)
- ✨ **Persistent Track IDs**: STATE output maintains IDs across frames for temporal effects
- ✨ **Resolution-Adaptive**: Distance thresholds scale with image size (4K/8K ready)
- ✨ **Simplified Workflow**: 3 parameters vs 15+ in Object/Motion tracker

**When to use:**
- ✅ Need color-based tracking (hue boundaries, saturation)
- ✅ Want preset-based workflow (no CV knowledge required)
- ✅ Need persistent track IDs for temporal effects
- ✅ Artistic VFX, video editing, creative workflows
- ✅ Fast batch video processing (optimized for speed)

**GPU Performance:** 80% coverage (4/5 modes fully GPU-accelerated)

---

### Which Tracker to Use?

| Need | Use This Tracker |
|------|------------------|
| Track objects (people, cars) | **Object/Motion** (YOLO detection) |
| Track motion between frames | **Object/Motion** (optical flow) |
| Visual attention/saliency | **Object/Motion** (saliency_map) |
| Fine-tune CV algorithms | **Object/Motion** (15+ parameters) |
| Track color boundaries | **Colors/Luma** ✨ (Color Hunter mode) |
| Track colorful regions | **Colors/Luma** ✨ (Chroma Density mode) |
| Persistent track IDs | **Colors/Luma** (STATE output) |
| Fastest GPU performance | **Colors/Luma** (80% GPU, simpler) |
| Beginner-friendly | **Colors/Luma** (presets) |
| Expert control | **Object/Motion** (methods) |

**Pro Tip:** Start with **Colors/Luma** for quick artistic results, switch to **Object/Motion** if you need YOLO, optical flow, or saliency detection.

---

### Track Manipulation Nodes (6 nodes)

**YS Track Merge** - Combines multiple track sources into unified output
**Inputs:** 2-4 track sources, deduplication options
**Outputs:** Merged tracks, total count
**Features:** Batch/video support, optional deduplication, distance threshold control

**YS Track Deduplicate** - Removes overlapping/clustered points
**Inputs:** Tracks, min distance, keep strategy (first/last/center/random), max points
**Outputs:** Filtered tracks, statistics (input/output counts, reduction %)
**Features:** GPU-accelerated (100× faster), batch processing, multiple keep strategies
**GPU Performance:** Distance matrix computation fully GPU-accelerated

**YS Track Jitter** - Applies jitter and push-apart effects
**Inputs:** Tracks, push_apart distance, jitter amount, iterations, seed
**Outputs:** Modified tracks
**Features:** GPU-accelerated distance calculations (10-50× faster), deterministic per-point offsets, stable across frames
**Effects:** Push-apart (separates overlapping points), Jitter (consistent random variation)

---

### Rendering Nodes (9 nodes)

**Line Link Renderer (Advanced)** - Renders mathematical curves connecting tracked points
**Inputs:** Tracks, dimensions, curve type, line style, parameters
**Outputs:** RGBA layer
**Curve types:** straight, bezier_quadratic, bezier_cubic, catmull_rom, logarithmic_spiral, elastic_curve, fourier_series, field_line, gravitational_path, voronoi_edges, delaunay_triangulation, minimum_spanning_tree, neural_flow_field
**Line styles:** solid, dotted, dashed, dash_dot, gradient_fade, pulsing, electric, lightning, particle_trail, wave_modulation

**Dot Renderer** - Renders tracked points as styled markers
**Inputs:** Tracks, dimensions, size, style, glow
**Outputs:** RGBA layer
**Styles:** solid, ring, cross, plus, square, diamond

**BBox Renderer** - Renders bounding boxes around tracked points
**Inputs:** Tracks/boxes, dimensions, box mode (fixed/radius/age), stroke width, fill opacity, roundness, color
**Outputs:** RGBA layer
**Features:** GPU-accelerated SDF rendering (2000× faster @ 4K), color picker support, anti-aliased rounded corners
**GPU Performance:** Custom CUDA kernels for distance field computation

**Blur Region Renderer** - Applies blur effects to regions around tracked points
**Inputs:** Image, tracks/boxes, radius, blur sigma, falloff, opacity
**Outputs:** LAYER (blurred regions)
**Features:** GPU-accelerated Gaussian blur (10-50× faster), smooth sigmoid falloff, depth-of-field effects
**GPU Performance:** CuPy Gaussian filtering

**MV Look Renderer** - Applies machine vision/surveillance camera aesthetic
**Inputs:** Image, scanline intensity, chroma offset, vignette, noise, as_layer option
**Outputs:** Modified image or overlay layer
**Features:** CRT scanlines, chromatic aberration, vignette, film grain, color tinting (surveillance green)
**Effects:** Retro technical/surveillance looks

**YS Echo Layer** - Temporal decay/trails effect
**Inputs:** Layer, decay mode, decay factor, boost, max age, exposure, clamp mode
**Outputs:** Layer with trails, internal state
**Features:** GPU-accelerated with custom CUDA kernel (3-5× faster), exponential moving average, optical flow warping support
**Modes:** accumulate, pingpong, framecount
**GPU Performance:** Custom CUDA EMA update kernel

**YS Pixel Sorting (Tracks)** - Track-gated pixel sorting glitch effect
**Inputs:** Image, tracks/boxes/mask, region mode, direction, threshold mode, sort order
**Outputs:** Processed image
**Features:** GPU-accelerated mask/metric computation, multiple sorting directions (h/v/radial/tangent), flexible threshold modes (luma/sat/hue/sobel)
**Directions:** horizontal, vertical, radial, tangent_flow
**Threshold modes:** luma, saturation, hue, sobel_mag

**YS Text On Tracks** - Renders text labels at tracked positions
**Inputs:** Tracks, dimensions, content mode, font, size, color, stroke, alignment
**Outputs:** RGBA layer with text
**Content modes:** coordinates (show [x,y]), code_snippets (random code lines), string (custom text)
**Features:** PIL/Pillow rendering, full typography control, kerning support, font discovery, batch processing
**Typography:** Font selection, size, letter spacing, stroke width, alignment (9 positions)

---

### Utility Nodes (4 nodes)

**Layer Merge** - Composites multiple RGBA layers
**Inputs:** 2-4 layers, blend mode, opacity
**Outputs:** Merged layer
**Modes:** normal, add, multiply, screen, overlay, max, min

**Composite Over** - Alpha-composites layer onto base image
**Inputs:** Base image, layer, opacity, offset
**Outputs:** Composited image

**Image Size Detector** - Extracts dimensions from image tensors
**Inputs:** Image
**Outputs:** Width, height as integers
**Use case:** Feed dimensions to other nodes automatically

**Video Frame Offset** - Offsets video frames by N frames
**Inputs:** Image batch, offset amount
**Outputs:** Offset image batch
**Use case:** Temporal alignment, create frame delays

## Example Workflow

**Typical configuration:**
- Detection: `2D Tracker (Colors/Luma)` with `Exploratory Luma` mode, 500 points
- Curves: `Line Link Renderer` with `logarithmic_spiral` and `electric` style
- Dots: `Dot Renderer` with `ring` style and 8.0 glow
- Blend: `Layer Merge` with `add` mode

## Performance

Benchmarks on RTX 5090 with GPU acceleration:

| Resolution | Object/Motion Tracker | Colors/Luma Tracker | Curves | Render | Total  | Target FPS |
|------------|-----------------------|---------------------|--------|--------|--------|------------|
| 1080p      | 3-8ms                 | 2-4ms               | <2ms   | <3ms   | <10ms  | 100+       |
| 4K         | 6-15ms                | 4-8ms               | <5ms   | <8ms   | <23ms  | 60         |
| 8K         | 15-35ms               | 10-20ms             | <10ms  | <20ms  | <55ms  | 18+        |

**Note:** optical_flow in Object/Motion tracker is slower (35-50ms @ 4K) due to CPU-only limitation.

## Detection Method Guide

### Object/Motion Tracker Methods

**gradient_magnitude** - Scharr gradient edge detection. Fully GPU-accelerated, fast, works on high-texture images.  
**phase_congruency** - FFT-based frequency domain features. Fully GPU-accelerated, lighting-invariant, excellent for patterns.  
**structure_tensor** - Shi-Tomasi corner detection. Fully GPU-accelerated, identifies keypoints and junctions.  
**optical_flow** - Farneback dense motion tracking. ⚠️ CPU-only (OpenCV limitation), requires previous_frame input, slower but tracks motion.  
**saliency_map** - Spectral residual visual attention. Fully GPU-accelerated (CuPy FFT), highlights perceptually salient regions.  
**object_detection** - YOLO v8 semantic detection. Fully GPU-accelerated (native YOLO), tracks specific object classes (requires ultralytics).  
**hybrid_adaptive** - Combines gradient + structure tensor. Fully GPU-accelerated, best overall accuracy, higher compute cost.

### Colors/Luma Tracker Modes

**Exploratory Luma** - Tracks edges and brightness changes using gradient magnitude. General-purpose, fast.  
**Color Hunter** - Tracks hue boundaries and color discontinuities. ✨ Unique to Colors/Luma tracker, weighted by saturation.  
**Locked Corners** - Tracks Harris corners where edges intersect. Best for geometric features.  
**Chroma Density** - Tracks high-saturation, colorful regions. ✨ Unique to Colors/Luma tracker, saturation × brightness.  
**Phase Congruency** - Multi-scale edge detection. Detects edges at multiple blur levels (σ = 1, 2, 4).

## Documentation

### User Guides
- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[Node Catalog](docs/NODE_CATALOG.md)** - Complete reference for all 21 nodes

### Developer Guides
- **[System Architecture](docs/SYSTEM_ARCHITECTURE.md)** - Node architecture, data types, batch processing
- **[GPU Acceleration](docs/GPU_ACCELERATION.md)** - GPU implementation patterns and performance
- **[Color Picker Implementation](docs/COLOR-PICKER-IMPLEMENTATION.md)** - Color input standardization
- **[Text Rendering Implementation](docs/TEXT-RENDERING-IMPLEMENTATION.md)** - Text rendering patterns
- **[Project Rules](CLAUDE.md)** - Development guidelines and critical rules

## Development

```bash
# Unit tests
pytest tests/unit/

# Visual regression tests
python tests/visual/generate_references.py
python tests/visual/compare_outputs.py

# GPU benchmarks
python tests/performance/benchmark_gpu.py
```

**Project status:**
- Phase 1 (complete): Core tracking and curve rendering
- Phase 2 (in progress): Extended renderers (bbox, blur, HUD, MV-look)
- Phase 3 (planned): Optimization passes
- Phase 4 (planned): Neural-guided features

## Technical Notes

GPU memory automatically configured for 8GB per operation (safe limit for 24GB VRAM). PyTorch uses 80% available memory. TensorFloat32 enabled by default.

All curve implementations use proper mathematical equations, not approximations. Rendering uses anti-aliased line drawing with sub-pixel accuracy.

## License

MIT License - see LICENSE file

## Credits

Developed by Yambo Studio for ComfyUI  
Optimized for NVIDIA RTX 5090

---

**Support:** [GitHub Issues](https://github.com/yamb0x/comfyui-ys-vision-tools/issues) | **Documentation:** [docs/plans/](docs/plans/)