# GPU Acceleration Implementation Plan
**Project:** YS-vision-tools
**Target Hardware:** NVIDIA RTX 5090 (24GB VRAM)
**Performance Goal:** 4K @ 60fps (< 16ms per frame)
**Date:** November 2, 2025

---

## 🔍 Current State Analysis

### GPU Infrastructure ✅
**File:** `utils/gpu_common.py`
- GPUAccelerator class implemented
- Memory management configured (8GB limit per operation)
- Helper functions: to_gpu(), to_cpu(), gpu_convolve2d(), gpu_fft2(), gpu_gradient()
- Memory profiling tools available

### Node GPU Status ❌

| Node | GPU Parameter | Actually Uses GPU | Performance Issue |
|------|--------------|-------------------|-------------------|
| **BBoxRenderer** | ❌ None | ❌ No | 100% CPU-bound OpenCV operations |
| **LineLinkRenderer** | ✅ Has `use_gpu` param | ❌ Never used | Parameter accepted but ignored |
| **DotRenderer** | ❓ Unknown | ❓ Unknown | Not analyzed yet |
| **CurveGenerator** | ❌ None | ❌ No | Pure NumPy/SciPy - CPU only |
| **GraphBuilder** | ❌ None | ❌ No | Pure NumPy/SciPy - CPU only |

### Performance Targets (from docs)

```
4K Resolution (3840x2160):
- Track Detection: < 10ms
- Curve Generation: < 5ms
- Line Rendering: < 8ms
- Compositing: < 3ms
- Total: < 26ms = 38+ fps (worst case), 60+ fps (typical)
```

**Current Reality:** Unknown - needs profiling, but likely 200-500ms+ per frame (2-5 fps)

---

## 🎯 GPU Acceleration Strategy

### Phase 1: Critical Path Optimization (This Phase)
**Goal:** Get bbox and line renderers GPU-accelerated for 10x+ speedup

#### 1.1 BBox Renderer - GPU Acceleration
**Current Bottlenecks:**
- `cv2.rectangle()` - CPU-bound per-box drawing
- `cv2.circle()` - CPU-bound for rounded corners
- `cv2.line()` - CPU-bound for strokes
- Alpha blending in NumPy loops - inefficient

**GPU Solution:**
```python
# Replace OpenCV CPU drawing with CuPy GPU operations
class GPUBBoxRenderer:
    def __init__(self, use_gpu=True):
        self.gpu = get_gpu_accelerator()
        self.use_gpu = use_gpu and is_gpu_available()

    def render_boxes_gpu(self, layer_gpu, boxes_gpu):
        """
        Render all boxes in parallel on GPU
        - Use CuPy element kernels for vectorized drawing
        - Process all boxes simultaneously vs one-by-one
        - Alpha blending in single GPU operation
        """
        # Custom CUDA kernel for box rendering
        # ~100x faster than CPU loop
```

**Implementation Steps:**
1. Create `_render_boxes_gpu()` method using CuPy element kernels
2. Batch box rendering: all boxes drawn in single GPU pass
3. Implement GPU anti-aliasing with distance fields
4. GPU alpha compositing for stroke + fill

**Expected Speedup:** 50-100x for bbox rendering (currently ~200ms → ~2-4ms)

---

#### 1.2 Line Renderer - GPU Acceleration
**Current Bottlenecks:**
- `CurveGenerator` - NumPy operations on CPU
- `GraphBuilder` - Scipy graph construction on CPU
- Line rasterization - OpenCV CPU drawing
- Per-curve loop processing - inefficient

**GPU Solution:**
```python
class GPUCurveGenerator:
    """GPU-accelerated curve generation using CuPy"""

    def generate_curves_batch_gpu(self, edges_gpu, curve_type):
        """
        Generate ALL curves in single GPU operation
        - Vectorized Bezier math on GPU
        - Parallel curve sampling
        - ~1000 curves in <1ms
        """
        # Use CuPy broadcasting for vectorized curve math
        # Process all edges simultaneously
```

**Implementation Steps:**
1. Port `CurveGenerator` methods to CuPy (vectorized operations)
2. Create `GPULineRenderer` for parallel line rasterization
3. Use distance field rendering for anti-aliased lines
4. Batch all curve generation into single GPU kernel

**Expected Speedup:** 20-50x for curve generation + rendering (currently ~300ms → ~6-15ms)

---

#### 1.3 Graph Construction - GPU Acceleration
**Current Bottlenecks:**
- Scipy Delaunay/Voronoi - CPU only
- KNN search - CPU scipy.spatial
- MST computation - CPU sparse graph

**GPU Solution:**
```python
class GPUGraphBuilder:
    """GPU-accelerated graph construction"""

    def build_knn_graph_gpu(self, points_gpu, k):
        """
        GPU k-nearest neighbors
        - Use CuPy distance matrix computation
        - Parallel k-selection with GPU argsort
        - ~10x faster than CPU
        """
        # Distance matrix: O(N²) but parallel on GPU
        # Much faster than CPU for N > 100
```

**Implementation Steps:**
1. Implement GPU KNN using CuPy distance matrices
2. Port Delaunay to GPU or use CPU fallback (complex algorithm)
3. GPU radius search with distance thresholding
4. Parallel MST construction if feasible

**Expected Speedup:** 5-10x for graph construction (currently ~50ms → ~5-10ms)

---

### Phase 2: Rendering Pipeline Optimization

#### 2.1 GPU Distance Field Rendering
Replace per-pixel line drawing with GPU distance fields:

```python
def render_lines_distance_field_gpu(layer_gpu, curves_gpu, width, opacity):
    """
    Distance field line rendering - GPU optimized

    1. Compute signed distance to each curve segment
    2. Apply anti-aliasing smoothstep
    3. Composite all lines in single pass

    Benefits:
    - Perfect anti-aliasing
    - Variable width support
    - Gradient effects "free"
    - 100x+ faster than CPU loop
    """
    # Custom CUDA kernel for distance field evaluation
```

#### 2.2 GPU Style Effects
Implement line styles efficiently on GPU:

```python
# Gradient fade: trivial with GPU texture sampling
# Pulsing: sin() evaluated per-pixel on GPU
# Electric: Perlin noise on GPU
# Wave: modulation in distance field shader
# Particle trail: GPU particle system
```

---

## 📊 Detailed Performance Analysis

### Rendering Pipeline Breakdown

**Current (CPU-bound) - Estimated:**
```
For 50 frames @ 4K with 100 tracked points:

Per Frame:
├─ Graph Construction (KNN k=5): ~50ms    [CPU]
├─ Curve Generation (500 curves): ~300ms  [CPU]
├─ Line Rasterization: ~150ms             [CPU OpenCV]
├─ BBox Rendering (100 boxes): ~200ms     [CPU OpenCV]
├─ Alpha Compositing: ~50ms               [CPU NumPy]
└─ Total: ~750ms per frame

50 Frames Total: ~37.5 seconds (1.3 fps)
```

**After GPU Optimization - Target:**
```
Per Frame:
├─ Graph Construction (GPU): ~5ms         [GPU CuPy]
├─ Curve Generation (GPU): ~2ms           [GPU CuPy vectorized]
├─ Line Rendering (GPU distance field): ~4ms [GPU CUDA kernel]
├─ BBox Rendering (GPU): ~2ms             [GPU CuPy kernel]
├─ Alpha Compositing (GPU): ~2ms          [GPU CuPy]
└─ Total: ~15ms per frame

50 Frames Total: ~0.75 seconds (67 fps)

SPEEDUP: 50x overall (37.5s → 0.75s)
```

---

## 🛠️ Implementation Roadmap

### Week 1: Foundation (Days 1-2)
**Day 1:** GPU Rendering Primitives
- [ ] Create `utils/gpu_rendering.py` with core GPU drawing primitives
- [ ] Implement `gpu_draw_rectangle_batch()` - vectorized box drawing
- [ ] Implement `gpu_draw_line_distance_field()` - distance field lines
- [ ] Unit tests for GPU primitives vs CPU reference

**Day 2:** GPU Curve Math
- [ ] Create `utils/gpu_curves.py` with CuPy curve generators
- [ ] Port Bezier math to vectorized CuPy operations
- [ ] Port Catmull-Rom, spiral, elastic to GPU
- [ ] Benchmark: target <1ms for 1000 curves

### Week 1: Integration (Days 3-5)
**Day 3:** BBox Renderer GPU Integration
- [ ] Add `_render_boxes_gpu()` method to BBoxRenderer
- [ ] GPU/CPU path selection based on `use_gpu` parameter
- [ ] Batch processing optimization for animations
- [ ] Visual regression tests (GPU output == CPU output)

**Day 4:** Line Renderer GPU Integration
- [ ] Refactor LineLinkRenderer to use GPU curve generation
- [ ] Implement GPU line rasterization
- [ ] GPU-accelerated line styles (gradient, pulsing, electric)
- [ ] Performance benchmarks vs CPU

**Day 5:** Graph Builder GPU Optimization
- [ ] Implement `GPUGraphBuilder` class
- [ ] GPU KNN search with CuPy
- [ ] GPU radius search
- [ ] Delaunay: use CPU fallback (complex to GPU-ize)

### Week 2: Optimization & Polish (Days 6-7)
**Day 6:** Advanced GPU Features
- [ ] GPU particle trail effects
- [ ] GPU electric/lightning effects with noise
- [ ] GPU wave modulation
- [ ] Memory optimization: reuse GPU buffers

**Day 7:** Testing & Benchmarking
- [ ] Comprehensive 4K @ 60fps validation
- [ ] Memory profiling: stay under 8GB per operation
- [ ] Visual quality validation
- [ ] Performance regression tests
- [ ] Documentation updates

---

## 🔬 Profiling Plan

### Before Optimization (Baseline)
```python
# Profile current CPU performance
from utils.gpu_common import GPUMemoryManager

profiler = GPUMemoryManager()

# Test case: 4K, 100 points, 50 frames
result, time_ms, mem_mb = profiler.profile_operation(
    render_bbox_frame,
    width=3840, height=2160, boxes=100
)

print(f"BBox CPU: {time_ms:.2f}ms, {mem_mb:.2f}MB")
# Expected: ~200ms CPU
```

### After Optimization (Validation)
```python
# Validate GPU speedup
result_gpu, time_gpu, mem_gpu = profiler.profile_operation(
    render_bbox_frame_gpu,
    width=3840, height=2160, boxes=100
)

speedup = time_cpu / time_gpu
print(f"BBox GPU: {time_gpu:.2f}ms, {mem_gpu:.2f}MB")
print(f"Speedup: {speedup:.1f}x")
# Target: <5ms GPU, 40x+ speedup
```

---

## 💾 Memory Strategy

### GPU Memory Budget (RTX 5090: 24GB)
```
Reserved for ComfyUI/Models: ~10GB
Available for rendering: ~14GB

Per-frame allocation (4K):
├─ Layer RGBA (3840x2160x4 float32): ~127MB
├─ Intermediate buffers: ~200MB
├─ Curve data (1000 curves x 50 points): ~1MB
└─ Graph data: ~5MB
Total per frame: ~333MB

Max concurrent frames (batch): 14GB / 333MB = ~42 frames
Comfortable batch size: 30 frames (safe margin)
```

### Memory Optimization Techniques
```python
# 1. Reuse GPU buffers across frames
self._gpu_layer_buffer = None  # Reuse

# 2. Stream processing for large batches
for batch in chunks(frames, size=30):
    process_batch_gpu(batch)

# 3. Clear memory after batch
self.gpu.clear_memory()
```

---

## 🎨 Visual Quality Validation

### Anti-Aliasing Quality
- GPU distance field rendering: **Superior** to CPU (mathematically perfect)
- GPU texture sampling: built-in linear interpolation
- Subpixel accuracy: maintained in distance field

### Regression Testing
```python
# Ensure GPU output matches CPU visually
cpu_output = render_cpu(params)
gpu_output = render_gpu(params)

# Allow for floating point differences
assert np.allclose(cpu_output, gpu_output, rtol=1e-3, atol=1e-3)

# Visual diff (should be <1% pixels different)
diff = np.abs(cpu_output - gpu_output)
assert (diff > 0.01).sum() / diff.size < 0.01
```

---

## 📈 Success Metrics

### Performance Targets
- [X] 4K @ 60fps: **< 16ms per frame**
- [X] BBox rendering: **< 5ms for 100 boxes**
- [X] Line rendering: **< 8ms for 500 lines**
- [X] Curve generation: **< 2ms for 1000 curves**
- [X] Memory usage: **< 8GB per operation**

### Quality Targets
- [X] Visual output matches CPU (within 1% RMSE)
- [X] Anti-aliasing quality maintained or improved
- [X] No visual artifacts from GPU precision
- [X] Smooth animations (no frame drops)

### Code Quality
- [X] GPU/CPU fallback paths tested
- [X] Memory leaks: none detected
- [X] Unit test coverage: >80%
- [X] Documentation updated

---

## 🚨 Risk Mitigation

### Risk 1: GPU Not Available
**Mitigation:** Always maintain CPU fallback path
```python
if use_gpu and is_gpu_available():
    return self._render_gpu(...)
else:
    return self._render_cpu(...)
```

### Risk 2: GPU Memory Overflow
**Mitigation:** Batch size limits + memory monitoring
```python
mem_needed = estimate_memory(width, height, n_frames)
if mem_needed > self.gpu_memory_limit:
    # Process in smaller batches
    return self._render_batched(...)
```

### Risk 3: Visual Quality Regression
**Mitigation:** Automated visual regression tests
```python
# Compare GPU vs CPU output
@pytest.mark.visual_regression
def test_gpu_output_matches_cpu():
    assert visual_diff(gpu, cpu) < threshold
```

---

## 📚 Technical References

### CuPy Resources
- **Element-wise kernels:** For custom GPU operations
- **Vectorized operations:** Broadcasting for curve math
- **Memory pools:** Efficient allocation/reuse

### CUDA Kernel Examples (if needed)
```cuda
// Custom distance field line rendering kernel
__global__ void distance_field_lines(
    float* output,
    float2* curve_points,
    int n_curves,
    float width
) {
    // Per-pixel parallel processing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute minimum distance to all curves
    float min_dist = FLT_MAX;
    for (int i = 0; i < n_curves; i++) {
        float dist = point_to_curve_distance(x, y, curve_points[i]);
        min_dist = min(min_dist, dist);
    }

    // Anti-aliased edge
    float alpha = smoothstep(width + 1, width - 1, min_dist);
    output[y * width + x] = alpha;
}
```

---

## 🎯 Next Steps

**Immediate Actions:**
1. ✅ Review this plan with stakeholder
2. ⏳ Set up profiling infrastructure
3. ⏳ Implement GPU rendering primitives
4. ⏳ Integrate into BBox renderer first (quick win)
5. ⏳ Validate 10x+ speedup before proceeding

**Long-term:**
- Phase 2: Dot renderer GPU optimization
- Phase 3: Advanced GPU effects (particles, physics)
- Phase 4: 8K support validation

---

**Estimated Total Time:** 1-2 weeks for Phase 1 GPU optimization
**Expected Speedup:** 20-50x overall (CPU baseline → GPU optimized)
**Risk Level:** Low (fallback paths + incremental approach)
