# CLAUDE.md - YS-vision-tools Project Rules

## 🚀 Project: YS-vision-tools
**Description**: Advanced ComfyUI custom nodes for GPU-accelerated vision overlays with experimental mathematical curves and smart detection methods.

**Tech Stack**: Python 3.10+, ComfyUI, OpenCV, NumPy, CuPy (CUDA), PyTorch, Ultralytics (YOLO)

**Target Platform**: NVIDIA RTX 5090 (24GB VRAM) - 4K@60fps performance

---

## 🔴 CRITICAL: File Location Rules

**ALWAYS EDIT HERE** (Source of Truth):
```
D:\Yambo Studio Dropbox\AI\vibe_coding\comfyui-custom-nodes\custom_nodes\ys_vision_tools\
```

**NEVER EDIT HERE** (Active Installation - User Copies Manually):
```
F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\
```

**Workflow**:
1. Make ALL changes to `D:\Yambo Studio Dropbox\...` files
2. User copies to `F:\Comfy3D_WinPortable\...` manually
3. DO NOT edit F: drive files directly - they will be overwritten

---

## ⚡ EFFICIENCY RULES - READ THIS FIRST

### 🎯 Focus on Solutions, Not Documentation

**DO**:
- Fix bugs directly in code
- Add debug logging when needed
- Test fixes immediately
- Brief status updates only

**DON'T**:
- Write long .md reports unless explicitly requested
- Create multiple documentation files per session
- Generate verbose explanations when code speaks for itself
- Spend tokens on formatting when solving is needed

### 📝 Communication Style

- **Concise updates**: "Fixed X in file.py:123, testing now"
- **Code over words**: Show the fix, not a 500-line explanation
- **Ask questions**: When unclear, ask quickly - don't guess and document
- **Results matter**: Working code > perfect documentation

---

## 🔴 CRITICAL RULES - NEVER SKIP

### 1. ALWAYS Verify Code Changes Before Asking User to Test

**MANDATORY VERIFICATION BEFORE USER TESTING**:
```bash
# Before saying "copy this file and test":
grep -n "NEW_FEATURE_MARKER" "D:\path\to\file.py"  # Verify new code exists
wc -l "D:\path\to\file.py"                        # Check line count changed
```

**Why This Matters**:
- Edit tool can fail silently
- Asking user to test non-existent code wastes their time
- ALWAYS verify the actual code changes were written to D: drive

**DON'T**:
- ❌ Assume Edit tool worked without verification
- ❌ Ask user to test without checking file contents
- ❌ Trust tool output without grep/Read verification

**DO**:
- ✅ Use grep to confirm new code exists in file
- ✅ Check line count increased/changed as expected
- ✅ Read the actual section to verify changes
- ✅ Only then ask user to copy and test

### 2. GPU-First Architecture - ALWAYS

**MANDATORY for ALL new nodes and operations**:

Every computational operation MUST have GPU acceleration:
- Image processing → CuPy
- Distance calculations → GPU vectorized operations
- Mathematical operations → GPU arrays
- Filters/convolutions → cupyx.scipy

**Standard Pattern** (from blur_region_renderer.py):
```python
def process(data, use_gpu=True):
    """ALWAYS provide use_gpu parameter (default True)"""

    if use_gpu and CUPY_AVAILABLE:
        start_time = time.perf_counter()

        # Transfer to GPU
        data_gpu = cp.asarray(data)

        # Process on GPU
        result_gpu = gpu_operation(data_gpu)

        # Transfer back
        result = cp.asnumpy(result_gpu)

        print(f"[YS-NODE] GPU operation in {(time.perf_counter()-start_time)*1000:.2f}ms")
        return result
    else:
        # CPU fallback with timing
        if use_gpu and not CUPY_AVAILABLE:
            print("[YS-NODE] GPU requested but CuPy unavailable, using CPU")

        start_time = time.perf_counter()
        result = cpu_operation(data)
        print(f"[YS-NODE] CPU operation in {(time.perf_counter()-start_time)*1000:.2f}ms")
        return result
```

**Every operation MUST have**:
- ✅ `use_gpu` parameter (default True)
- ✅ GPU path with CuPy when available
- ✅ CPU fallback path
- ✅ Performance logging for BOTH paths
- ✅ Automatic CUPY_AVAILABLE detection

**Reference Implementations**:
- `blur_region_renderer.py` - GPU mask creation + GPU blur
- `bbox_renderer.py` - GPU SDF rendering with CUDA kernels
- `track_detect.py` - GPU gradient computation
- `line_link_renderer.py` - GPU graph building with FAISS-GPU

**Never write CPU-only code** - if you can't GPU accelerate immediately, document it as technical debt.

### 3. CRITICAL: Batch Processing for Video - ALWAYS

**MANDATORY for ALL nodes that process LAYER or IMAGE tensors**:

ComfyUI passes video as batched tensors with shape `(B, H, W, C)` where B is the number of frames.
**EVERY node MUST process ALL frames in the batch**, not just frame 0.

**🔴 CRITICAL BUG PATTERN TO AVOID**:
```python
# ❌ WRONG - Only processes first frame!
def execute(self, layer: torch.Tensor, ...):
    layer_np = layer[0].cpu().numpy()  # BUG: Only frame 0!
    result = process(layer_np)
    return (torch.from_numpy(result).unsqueeze(0),)
```

**✅ CORRECT PATTERN - Process all frames**:
```python
def execute(self, layer: torch.Tensor, ...):
    # 1. Detect batch size
    batch_size = layer.shape[0]
    is_batch = batch_size > 1

    if is_batch:
        print(f"[YS-NODE] BATCH MODE: {batch_size} frames")

    # 2. Initialize state (OUTSIDE loop for temporal nodes)
    if state is None:
        state = self._init_state(...)

    # 3. Process EACH frame
    output_frames = []
    for i in range(batch_size):
        # Get current frame
        frame_np = layer[i].cpu().numpy()  # Process frame i

        # Process frame (state persists for temporal effects!)
        result_np, state = self._process(frame_np, state, ...)

        output_frames.append(result_np)

        # Progress logging (every 10 frames)
        if is_batch and (i % 10 == 0 or i == batch_size - 1):
            print(f"[YS-NODE] Processed frame {i+1}/{batch_size}")

    # 4. Stack back into batch
    if is_batch:
        output_batch = np.stack(output_frames, axis=0)
        output_tensor = torch.from_numpy(output_batch).float()
    else:
        output_tensor = torch.from_numpy(output_frames[0]).unsqueeze(0).float()

    return (output_tensor, state)
```

**Key Rules**:
- ✅ **ALWAYS** loop through `range(batch_size)`, not just `[0]`
- ✅ **State persists** across frames (critical for temporal effects like Echo, trails, accumulation)
- ✅ **Stack output** with `np.stack(output_frames, axis=0)` to match input batch size
- ✅ **Log progress** every 10 frames to avoid console spam
- ✅ **Test with video** (50+ frames) not just single images

**Nodes That MUST Use Batch Processing**:
- ✅ **EchoLayer** - Temporal accumulation across frames
- ✅ **PixelSorting** - Per-frame sorting with animation
- ✅ **TextOnTracks** - Text labels on each frame
- ✅ **Any node** that takes IMAGE or LAYER as input

**Reference Implementation**: See `nodes/echo_layer.py:execute()` (lines 145-213)

**Common Mistakes**:
1. ❌ Only processing `layer[0]` → Effect frozen on first frame
2. ❌ Not stacking output → Returns single frame for 50-frame input
3. ❌ Reinitializing state inside loop → Temporal effects broken
4. ❌ No batch detection → Crashes on multi-frame input

**Verification Checklist**:
- [ ] Code has `for i in range(batch_size):` loop
- [ ] Each frame `layer[i]` is processed individually
- [ ] Output uses `np.stack(output_frames, axis=0)`
- [ ] Console shows `BATCH MODE: N frames` message
- [ ] Console shows progress `Processed frame X/N`
- [ ] Output tensor shape matches input: `(B, H, W, C)`

---

## 🎯 Project-Specific Guidelines

### GPU Development (RTX 5090)
```python
# ALWAYS provide GPU path with fallback
def process(data, use_gpu=True):
    if use_gpu and CUPY_AVAILABLE:
        return gpu_process(cp.asarray(data))
    return cpu_process(data)

# Memory management for 24GB VRAM
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=8 * 1024**3)  # Use 8GB max per operation
```

### GPU Acceleration Status

**Current GPU-Accelerated Nodes** (see `docs/plans/GPU-STATUS.md` for details):
- ✅ **BBox Renderer** - 2000× faster (CUDA SDF kernels)
- ✅ **Blur Renderer** - 10-50× faster (GPU mask + GPU blur)
- ✅ **Track Detect** - 5× faster (GPU gradient computation)
- ✅ **Line Graph Builder** - 25× faster (FAISS-GPU kNN)
- 🚧 **Line Rendering** - GPU graph done, line drawing CPU (Days 3-5)
- 🚧 **Dot Renderer** - CPU (GPU target: Day 10)

### GPU Performance Logging
All GPU-enabled nodes automatically log performance:
```
[YS-BBOX] GPU rendered 100 boxes @ 3840x2160 in 2.34ms
[YS-BBOX] CPU rendered 100 boxes @ 3840x2160 in 187.56ms
```

Enable/disable with `use_gpu` parameter in each node.

---

## 🧪 Testing Approach

### ⚠️ CRITICAL: Always Test in ComfyUI

**DO NOT use pytest or standalone Python tests for this project.**

**Why:**
- ComfyUI has its own Python environment with torch, cupy, etc.
- Standalone pytest runs in different environment → dependency hell
- Real-world bugs only show up in ComfyUI workflows
- Performance can only be measured in actual ComfyUI context

### Proper Testing Workflow

**1. Make Code Changes**
- Edit node files on D: drive (source of truth)
- Copy to F: drive ComfyUI installation

**2. Test in ComfyUI**
```bash
# Restart ComfyUI to load changes
cd F:\Comfy3D_WinPortable
# Start ComfyUI server
```

**3. Create Test Workflow**
- Add the node you changed
- Connect test data (images, tracks, etc.)
- Set parameters to test
- **Run and observe console output**

**4. Verify Results**
- Check console logs for performance: `[YS-BBOX] GPU rendered...`
- Inspect visual output quality
- Test edge cases (empty inputs, large batches, etc.)
- Verify GPU/CPU fallback works

**5. Test Edge Cases in ComfyUI**
- Empty input (no tracks)
- Single item
- Large batch (1000+ items)
- 4K resolution
- Animation (50+ frames)
- GPU disabled (`use_gpu=False`)

### Example Testing Session

```
1. Edit bbox_renderer.py on D: drive
2. Copy to F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\nodes\
3. Restart ComfyUI
4. Load workflow with BBox Renderer node
5. Run with 100 boxes @ 4K
6. Check console:
   [YS-BBOX] GPU rendered 100 boxes @ 3840x2160 in 2.34ms ✓
7. Disable GPU, rerun:
   [YS-BBOX] CPU rendered 100 boxes @ 3840x2160 in 187.56ms ✓
8. Verify visual output matches
```

### When Tests Fail

**In ComfyUI console, look for:**
- Error stack traces → fix the bug
- Performance logs → optimize if too slow
- GPU fallback warnings → check GPU availability

**DO NOT:**
- Try to fix by writing pytest tests
- Run standalone Python scripts
- Use different Python environment

**DO:**
- Fix in D: drive code
- Copy to F: drive
- Restart ComfyUI
- Test again

### Visual Testing Checklist

For renderer nodes, always verify:
- [ ] Output looks correct visually
- [ ] No artifacts or glitches
- [ ] Anti-aliasing quality good
- [ ] Colors match expected
- [ ] Transparency/alpha blending correct
- [ ] Performance acceptable (check console)

---

### ComfyUI Node Structure
```python
class YourNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}}

    RETURN_TYPES = ("TYPE",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Category"

    def execute(self, **kwargs):
        # Implementation
        pass
```

### Mathematical Rigor
```python
# Implement REAL equations, not approximations
def logarithmic_spiral(t, a=1, b=0.2):
    """r = a * exp(b * theta) - Proper mathematical implementation"""
    theta = t * 2 * np.pi
    r = a * np.exp(b * theta)
    return r * np.cos(theta), r * np.sin(theta)
```

### Performance Requirements
- **4K @ 60fps**: Total processing time < 16ms
- **Detection**: < 10ms
- **Curve Generation**: < 5ms
- **Rendering**: < 8ms
- **Memory**: < 8GB VRAM @ 4K

---

## 🧪 Testing Approach

### Visual Testing
```python
# Save visual outputs for inspection
def test_curve_rendering():
    layer = render_curve("logarithmic_spiral", params)
    save_test_image(layer, "test_spiral.png")

    # Automated checks
    assert layer.shape == (2160, 3840, 4)  # 4K RGBA
    assert layer.max() > 0  # Has content
    assert np.all(layer >= 0) and np.all(layer <= 1)  # Valid range
```

### GPU Performance Testing
```python
# Profile GPU operations
def benchmark_gpu_operation(func, iterations=100):
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iterations):
        func()
    end.record()
    end.synchronize()

    ms_per_op = cp.cuda.get_elapsed_time(start, end) / iterations
    assert ms_per_op < 10, f"Too slow: {ms_per_op}ms, need <10ms"
```

### Mathematical Validation
```python
# Verify curve equations
def test_curve_mathematics():
    # Test known values
    point = bezier_cubic(t=0.5, p0, p1, c1, c2)
    expected = calculate_expected_bezier_point(0.5)
    assert np.allclose(point, expected, rtol=1e-5)
```

---

## 📁 Project Structure

```
custom_nodes/
└── ys_vision_tools/           # Main package (renamed from ys_vision)
    ├── __init__.py            # Node registration
    ├── nodes/
    │   ├── track_detect.py    # Smart detection (7+ methods)
    │   ├── line_link_renderer.py  # Advanced curves (15+ types)
    │   ├── dot_renderer.py
    │   ├── palette_map.py
    │   ├── layer_merge.py
    │   └── composite_over.py
    ├── utils/
    │   ├── gpu_common.py      # GPU acceleration utilities
    │   ├── curve_math.py      # Mathematical curve functions
    │   └── image_utils.py     # Format conversions
    └── tests/
        ├── unit/              # Function tests
        ├── visual/            # Visual regression tests
        └── performance/       # GPU benchmarks
```

---

## 🔧 Development Workflow

### Adding a New Detection Method
1. Add to `DETECTION_METHODS` enum in `track_detect.py`
2. Implement `_detect_METHOD_NAME()` with GPU path
3. Add tests in `tests/unit/test_track_detect.py`
4. Benchmark at 4K resolution
5. Update documentation

### Adding a New Curve Type
1. Add to `CURVE_TYPES` enum in `line_link_renderer.py`
2. Implement mathematical equation in `_generate_curve()`
3. Add GPU version in `_generate_curve_gpu()` if complex
4. Create visual test in `tests/visual/`
5. Verify mathematical correctness

### Performance Optimization Workflow
1. Profile with `cProfile` or GPU profiler
2. Identify bottlenecks (usually CPU-GPU transfers)
3. Optimize with CuPy/CUDA kernels
4. Verify 4K@60fps target maintained
5. Document optimization in code comments

---

## ⚠️ Common Pitfalls to Avoid

### Image Format Issues
```python
# WRONG - Format confusion
image = torch_tensor  # (B, C, H, W)
cv2.process(image)   # Expects (H, W, C) numpy

# RIGHT - Explicit conversion
image = ensure_numpy_hwc(torch_tensor)
result = cv2.process(image)
```

### GPU Memory Leaks
```python
# WRONG - Memory accumulation
for frame in video:
    gpu_array = cp.asarray(frame)  # Keeps allocating

# RIGHT - Reuse memory
gpu_array = cp.empty(shape)
for frame in video:
    gpu_array[:] = frame  # Reuse allocation
```

### Mathematical Accuracy
```python
# WRONG - Approximation
curve = straight_line * (1 + random_offset)

# RIGHT - Proper equation
curve = catmull_rom_spline(points, tension=0.5)
```

---

## 📊 Success Metrics

### Phase 1 Complete When:
- [ ] All 7 detection methods implemented
- [ ] All 15 curve types rendering correctly
- [ ] 4K @ 60fps achieved on RTX 5090
- [ ] GPU memory usage < 8GB
- [ ] Visual tests passing
- [ ] Mathematical tests passing
- [ ] No CPU bottlenecks

### Quality Standards:
- Code coverage > 80%
- All GPU paths tested
- Visual outputs verified
- Mathematical correctness validated
- Performance benchmarks documented

---

## 🎨 Color Picker Standard Pattern

### ALWAYS Use ComfyUI COLOR Type

For ALL nodes with color parameters, use ComfyUI's native `COLOR` input type:

```python
# Import color utility
from ..utils import normalize_color_to_rgba01

# INPUT_TYPES - Move color to required, alpha to optional
"required": {
    "color": ("COLOR", {
        "default": "#ffffff",
        "tooltip": "Click the color swatch to open the visual color picker"
    }),
},
"optional": {
    "alpha": ("FLOAT", {
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "tooltip": "Transparency level (0=invisible, 1=opaque)"
    }),
}

# execute() - Include color in signature
def execute(self, ..., color, **kwargs):
    # Parse color (handles HEX, named colors, legacy formats)
    alpha = kwargs.get('alpha', 1.0)
    rgba = normalize_color_to_rgba01(color, alpha)
    color_rgb = rgba[:3]  # RGB tuple for rendering

    print(f"[YS-NODE] Parsed color: {color} -> RGBA: {rgba}")
    # Use color_rgb in rendering
```

### Key Rules:

1. **ALWAYS** use `("COLOR", ...)` input type, not `("STRING", ...)`
2. **ALWAYS** import `normalize_color_to_rgba01` from utils
3. **ALWAYS** provide HEX defaults: `"#ffffff"`, `"#ff0000"`, etc.
4. **ALWAYS** add separate `alpha` slider for transparency
5. **ALWAYS** parse color with `normalize_color_to_rgba01(color, alpha)`
6. **ALWAYS** extract RGB with `rgba[:3]` for rendering operations
7. **ALWAYS** add debug logging: `print(f"[YS-NODE] Parsed color: ...")`
8. **NEVER** parse color per-frame in batch loops (parse once in method)

### Supported Formats:

- **HEX**: `"#ffffff"`, `"#ff0000"`, `"#00ff00"` (primary format)
- **Named**: `"red"`, `"blue"`, `"white"`, `"cyan"`, `"orange"`
- **Legacy**: `"1.0,0.5,0.0"`, `[1.0, 0.5, 0.0]` (backward compatibility)

### Multi-Color Nodes:

For nodes with multiple colors (text + stroke, gradients):

```python
"required": {
    "color": ("COLOR", {"default": "#ffffff"}),
    "stroke_color": ("COLOR", {"default": "#000000"}),
},

def execute(self, ..., color, stroke_color, opacity, **kwargs):
    text_rgba = normalize_color_to_rgba01(color, opacity)
    stroke_rgba = normalize_color_to_rgba01(stroke_color, 1.0)
```

### Reference Implementation:

- **Single Color**: `nodes/bbox_renderer.py` - Complete working example
- **Multi-Color**: `nodes/text_on_tracks.py` - Text + stroke colors
- **Utility**: `utils/color_utils.py` - Parsing logic
- **Documentation**: `docs/COLOR-PICKER-IMPLEMENTATION.md`

---

## 🚀 Quick Commands

```bash
# Setup environment
pip install cupy-cuda12x torch ultralytics opencv-python numpy scipy

# Run tests
pytest tests/ -v

# Benchmark GPU
python tests/performance/benchmark_gpu.py

# Visual tests
python tests/visual/generate_references.py
python tests/visual/compare_outputs.py

# Profile performance
python -m cProfile -o profile.stats nodes/track_detect.py
```

---

## 📚 Key Documentation

- **Development Plan**: `/docs/plans/README.md`
- **Phase 1 Tasks**: `/docs/plans/02-PHASE1-MVP.md`
- **Enhancement Details**: `/docs/plans/ENHANCEMENT-SUMMARY.md`
- **Testing Guide**: `/docs/plans/04-TESTING-GUIDE.md`
- **Common Issues**: `/docs/plans/05-COMMON-PITFALLS.md`

---

## 💡 Remember

1. **GPU First**: Every operation should have a CuPy path
2. **Mathematical Rigor**: Implement real equations, not approximations
3. **Visual Uniqueness**: Effects should look distinctive and artistic
4. **Performance Critical**: 4K@60fps is the minimum bar
5. **Test Everything**: Visual, mathematical, and performance tests

---

*Project: YS-vision-tools | Platform: RTX 5090 | Target: 4K@60fps | Style: Experimental VFX*