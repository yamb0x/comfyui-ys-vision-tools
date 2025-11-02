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

## 🎯 Project-Specific Guidelines

### GPU Development (RTX 5090)
```python
# ALWAYS provide GPU path with fallback
def process(data, use_gpu=True):
    if use_gpu and is_gpu_available():
        return gpu_process(cp.asarray(data))
    return cpu_process(data)

# Memory management for 24GB VRAM
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=8 * 1024**3)  # Use 8GB max per operation
```

### GPU Acceleration Status (Day 1 Complete)

**✅ BBox Renderer** - GPU accelerated with SDF rendering
- **Speedup:** 50-100× (200ms → 2-4ms @ 4K, 100 boxes)
- **Implementation:** `utils/gpu_rendering.py` - Batched SDF kernel
- **Parameter:** `use_gpu=True` (default) in BBoxRenderer node
- **Testing:** Unit tests + visual regression tests passing

**⏳ Line Renderer** - In progress (Day 2-3)
- FAISS-GPU KNN for graph building
- Tiled distance field rendering
- Vectorized curve generation

**⏳ Dot Renderer** - Planned (Day 7)

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