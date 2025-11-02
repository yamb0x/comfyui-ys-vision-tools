# Full GPU Implementation Plan - Production Grade RTX 5090
**Project:** YS-vision-tools
**Goal:** Complete GPU-accelerated rendering pipeline for 4K @ 60fps
**Approach:** No shortcuts, robust and solid toolkit
**Timeline:** 2-3 weeks for full implementation + testing

---

## 🎯 Vision: Complete GPU Pipeline

### End-to-End GPU Data Flow
```
Input (ComfyUI Tensor)
    ↓ [Zero-copy DLPack transfer]
GPU Memory (persistent buffers)
    ↓
[FAISS-GPU KNN] → Graph edges (GPU)
    ↓
[Vectorized Curve Gen] → All curve points (GPU)
    ↓
[Tiled Distance Field] → Line layer FP16 (GPU)
    ↓
[SDF Batch Bbox] → Bbox layer FP16 (GPU)
    ↓
[GPU Dot Renderer] → Dot layer FP16 (GPU)
    ↓
[One-Pass Merge] → Composited RGBA FP16 (GPU)
    ↓ [Zero-copy DLPack transfer]
Output (ComfyUI Tensor)
```

**Key Principle:** Data stays on GPU from input to output. Zero CPU transfers except final result.

---

## 📋 Complete Implementation Phases

### Phase 1: Foundation & Quick Wins (Week 1)
**Goal:** Get basic GPU pipeline working with immediate speedup

#### Day 1: GPU BBox Renderer Integration
**File:** `nodes/bbox_renderer.py`

**Tasks:**
- [ ] Import `GPUBBoxRenderer` from `utils.gpu_rendering`
- [ ] Add GPU/CPU path selection in `_render_single_frame()`
- [ ] Convert boxes to Nx7 array format: `[x, y, w, h, r, g, b]`
- [ ] Handle batch mode: process all frames on GPU, minimize transfers
- [ ] Add performance logging: log CPU vs GPU time
- [ ] Visual regression test: GPU output matches CPU
- [ ] Benchmark: 100 boxes @ 4K should be <5ms

**Success Criteria:**
- ✅ 50-100× speedup (200ms → 2-4ms)
- ✅ Visual output identical to CPU (within 1% RMSE)
- ✅ No GPU memory leaks
- ✅ CPU fallback works

---

#### Day 2: GPU Graph Builder Integration
**File:** `nodes/line_link_renderer.py`

**Tasks:**
- [ ] Import `GPUGraphBuilder` from `utils.gpu_graph`
- [ ] Add `delta_y_cap` parameter to INPUT_TYPES (optional float)
- [ ] Add `degree_cap` parameter to INPUT_TYPES (optional int)
- [ ] Replace `GraphBuilder` calls with `GPUGraphBuilder` when `use_gpu=True`
- [ ] Pass delta_y_cap and degree_cap to GPU graph builder
- [ ] Test with various graph modes: knn, radius, delaunay, mst, voronoi
- [ ] Benchmark: 1000 points, k=5 should be <2ms

**Success Criteria:**
- ✅ 10-30× speedup for KNN (15ms → 0.5ms)
- ✅ Delta-y cap prevents vertical connections
- ✅ Degree cap limits hub nodes
- ✅ All graph modes work correctly

---

#### Day 3: GPU Line Renderer (Straight Lines)
**File:** `nodes/line_link_renderer.py`

**Tasks:**
- [ ] Import `GPULineRenderer` from `utils.gpu_rendering`
- [ ] Add GPU path for straight line rendering
- [ ] Convert edges and points to GPU arrays
- [ ] Call `render_lines_tiled()` for GPU rendering
- [ ] Handle color parameter (parse RGB string to tuple)
- [ ] Test with various line widths and opacities
- [ ] Benchmark: 500 edges @ 4K should be <6ms

**Success Criteria:**
- ✅ 20-50× speedup (150ms → 3-6ms)
- ✅ Perfect anti-aliasing (better than CPU)
- ✅ Works with all graph modes
- ✅ Handles sparse and dense edge cases

---

#### Day 4-5: Vectorized Curve Generation
**New File:** `utils/gpu_curves.py`

**Full Implementation Required:**

```python
"""
GPU-accelerated curve generation using vectorized CuPy operations

All curves generated in single GPU operation - no loops
Supports all curve types from line_link_renderer
"""

import cupy as cp
import numpy as np
from typing import Tuple


class GPUCurveGenerator:
    """
    Vectorized curve generation on GPU

    Key optimization: Generate ALL curves simultaneously
    Input: Edge indices + points
    Output: (n_edges, samples, 2) curve points

    Performance: 1000 curves in <1ms
    """

    def __init__(self, samples_per_curve: int = 50):
        self.samples = samples_per_curve

    def generate_curves_batch(
        self,
        edges: cp.ndarray,  # (E, 2) int32
        points: cp.ndarray,  # (N, 2) float32
        curve_type: str,
        **params
    ) -> cp.ndarray:
        """
        Generate all curves in single GPU operation

        Returns:
            (E, samples, 2) curve points on GPU
        """
        if curve_type == "straight":
            return self._straight_batch(edges, points)
        elif curve_type == "quadratic_bezier":
            return self._quadratic_bezier_batch(edges, points, **params)
        elif curve_type == "cubic_bezier":
            return self._cubic_bezier_batch(edges, points, **params)
        elif curve_type == "catmull_rom":
            return self._catmull_rom_batch(edges, points, **params)
        elif curve_type == "logarithmic_spiral":
            return self._logarithmic_spiral_batch(edges, points, **params)
        elif curve_type == "elastic":
            return self._elastic_batch(edges, points, **params)
        elif curve_type == "fourier_series":
            return self._fourier_batch(edges, points, **params)
        elif curve_type == "field_lines":
            return self._field_lines_batch(edges, points, **params)
        elif curve_type == "gravitational":
            return self._gravitational_batch(edges, points, **params)
        else:
            return self._straight_batch(edges, points)

    def _straight_batch(self, edges, points):
        """Vectorized straight lines"""
        E = len(edges)
        p1 = points[edges[:, 0]]  # (E, 2)
        p2 = points[edges[:, 1]]  # (E, 2)

        t = cp.linspace(0, 1, self.samples)  # (samples,)
        t = t[None, :, None]  # (1, samples, 1)

        # Linear interpolation: p1 + t*(p2-p1)
        curves = p1[:, None, :] + t * (p2 - p1)[:, None, :]
        return curves  # (E, samples, 2)

    def _quadratic_bezier_batch(self, edges, points, overshoot=0.0, **kwargs):
        """Vectorized quadratic Bezier: (1-t)²p0 + 2(1-t)t·c + t²p1"""
        E = len(edges)
        p1 = points[edges[:, 0]]  # (E, 2)
        p2 = points[edges[:, 1]]  # (E, 2)

        # Compute control points (perpendicular to line)
        mid = (p1 + p2) / 2
        vec = p2 - p1
        normal = cp.stack([-vec[:, 1], vec[:, 0]], axis=1)  # (E, 2)
        norm_length = cp.linalg.norm(normal, axis=1, keepdims=True)
        normal = cp.where(norm_length > 0, normal / norm_length, 0)

        control = mid + normal * cp.linalg.norm(vec, axis=1, keepdims=True) * (0.3 + overshoot)

        # Bezier formula
        t = cp.linspace(0, 1, self.samples)[None, :, None]
        curves = (
            (1-t)**2 * p1[:, None, :] +
            2*(1-t)*t * control[:, None, :] +
            t**2 * p2[:, None, :]
        )
        return curves

    def _cubic_bezier_batch(self, edges, points, overshoot=0.0,
                           control_offset=0.3, **kwargs):
        """Vectorized cubic Bezier"""
        E = len(edges)
        p1 = points[edges[:, 0]]
        p2 = points[edges[:, 1]]

        vec = p2 - p1
        normal = cp.stack([-vec[:, 1], vec[:, 0]], axis=1)
        norm_length = cp.linalg.norm(normal, axis=1, keepdims=True)
        normal = cp.where(norm_length > 0, normal / norm_length, 0)

        c1 = p1 + vec * control_offset + normal * cp.linalg.norm(vec, axis=1, keepdims=True) * (0.2 + overshoot)
        c2 = p2 - vec * control_offset + normal * cp.linalg.norm(vec, axis=1, keepdims=True) * (0.2 - overshoot)

        t = cp.linspace(0, 1, self.samples)[None, :, None]
        curves = (
            (1-t)**3 * p1[:, None, :] +
            3*(1-t)**2*t * c1[:, None, :] +
            3*(1-t)*t**2 * c2[:, None, :] +
            t**3 * p2[:, None, :]
        )
        return curves

    def _catmull_rom_batch(self, edges, points, tension=0.5, **kwargs):
        """Vectorized Catmull-Rom spline"""
        # TODO: Implement vectorized Catmull-Rom
        # For now, fall back to cubic Bezier approximation
        return self._cubic_bezier_batch(edges, points, overshoot=0)

    def _logarithmic_spiral_batch(self, edges, points, spiral_turns=0.5, **kwargs):
        """Vectorized logarithmic spiral"""
        E = len(edges)
        p1 = points[edges[:, 0]]
        p2 = points[edges[:, 1]]

        # Spiral from p1 to p2
        vec = p2 - p1
        dist = cp.linalg.norm(vec, axis=1, keepdims=True)
        angle_end = cp.arctan2(vec[:, 1:2], vec[:, 0:1])

        # Spiral: r = a * exp(b * theta)
        t = cp.linspace(0, 1, self.samples)[None, :, None]
        theta = t * spiral_turns * 2 * cp.pi
        r = t * dist

        # Rotate to align with p1->p2
        x = r * cp.cos(theta + angle_end)
        y = r * cp.sin(theta + angle_end)

        curves = p1[:, None, :] + cp.concatenate([x, y], axis=2)
        return curves

    def _elastic_batch(self, edges, points, stiffness=0.5, **kwargs):
        """Vectorized elastic curve (damped oscillation)"""
        E = len(edges)
        p1 = points[edges[:, 0]]
        p2 = points[edges[:, 1]]

        vec = p2 - p1
        normal = cp.stack([-vec[:, 1], vec[:, 0]], axis=1)
        norm_length = cp.linalg.norm(normal, axis=1, keepdims=True)
        normal = cp.where(norm_length > 0, normal / norm_length, 0)

        t = cp.linspace(0, 1, self.samples)[None, :, None]

        # Damped oscillation perpendicular to line
        amplitude = cp.linalg.norm(vec, axis=1, keepdims=True) * 0.2
        offset = amplitude * cp.sin(t * cp.pi * 3) * cp.exp(-stiffness * t * 5)

        # Base linear path + oscillating offset
        curves = p1[:, None, :] + t * vec[:, None, :] + offset * normal[:, None, :]
        return curves

    def _fourier_batch(self, edges, points, wave_amplitude=5.0,
                      wave_frequency=0.1, **kwargs):
        """Vectorized Fourier series (wave)"""
        E = len(edges)
        p1 = points[edges[:, 0]]
        p2 = points[edges[:, 1]]

        vec = p2 - p1
        normal = cp.stack([-vec[:, 1], vec[:, 0]], axis=1)
        norm_length = cp.linalg.norm(normal, axis=1, keepdims=True)
        normal = cp.where(norm_length > 0, normal / norm_length, 0)

        t = cp.linspace(0, 1, self.samples)[None, :, None]

        # Sine wave perpendicular to line
        offset = wave_amplitude * cp.sin(t * 2 * cp.pi / wave_frequency)

        curves = p1[:, None, :] + t * vec[:, None, :] + offset * normal[:, None, :]
        return curves

    def _field_lines_batch(self, edges, points, field_strength=1.0, **kwargs):
        """Vectorized field lines (electromagnetic-like)"""
        # Similar to elastic but with different decay
        return self._elastic_batch(edges, points, stiffness=field_strength)

    def _gravitational_batch(self, edges, points, gravity_strength=0.1, **kwargs):
        """Vectorized gravitational curve (parabolic arc)"""
        E = len(edges)
        p1 = points[edges[:, 0]]
        p2 = points[edges[:, 1]]

        t = cp.linspace(0, 1, self.samples)[None, :, None]

        # Parabolic arc (simulate gravity pulling down)
        gravity_offset = cp.array([[0.0, 1.0]])  # Pull downward
        gravity_offset = gravity_offset[None, :, :]  # (1, 1, 2)

        # Parabolic term: g * t * (1-t)
        arc = gravity_strength * t * (1-t) * 100 * gravity_offset

        # Base linear path + parabolic arc
        curves = p1[:, None, :] + t * (p2 - p1)[:, None, :] + arc
        return curves
```

**Integration Tasks:**
- [ ] Implement all curve types vectorized
- [ ] Test each curve type matches CPU output
- [ ] Benchmark: 1000 curves should be <1ms
- [ ] Add to line renderer GPU path

---

### Phase 2: Advanced GPU Features (Week 2)

#### Day 6: Fused Curve-Gen + Raster Kernel
**New File:** `utils/gpu_rendering.py` (extend)

**Goal:** Eliminate read/write cycle between curve generation and rasterization

**CUDA Kernel:**
```cuda
// Fused: generate curve samples + rasterize in same kernel
// Data stays in registers/shared mem, never written to global memory

extern "C" __global__
void fused_curve_raster(
    const float* points,        // (N, 2)
    const int* edges,           // (E, 2)
    const int* tile_edge_lists,
    const int* tile_edge_counts,
    float* output,              // (H, W, 4)
    int curve_type,             // 0=straight, 1=bezier, etc.
    // ... curve params
    // ... raster params
) {
    // Each block handles one tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    // Load tile edges to shared memory
    __shared__ int tile_edges[MAX_EDGES_PER_TILE];
    __shared__ float2 p1[MAX_EDGES_PER_TILE];
    __shared__ float2 p2[MAX_EDGES_PER_TILE];

    // ... load edges in coalesced manner

    // Each thread handles multiple pixels in tile
    for (int px = threadIdx.x; px < TILE_SIZE; px += blockDim.x) {
        for (int py = threadIdx.y; py < TILE_SIZE; py += blockDim.y) {

            float min_dist = 1e6f;

            // For each edge in this tile
            for (int i = 0; i < n_tile_edges; i++) {
                // Generate curve samples IN REGISTERS
                float2 curve[SAMPLES_PER_CURVE];
                generate_curve_samples(
                    p1[i], p2[i], curve_type, curve
                );

                // Compute distance to curve
                float dist = distance_to_curve(px, py, curve, SAMPLES_PER_CURVE);
                min_dist = min(min_dist, dist);
            }

            // Write pixel
            if (min_dist < line_width) {
                // AA and write
            }
        }
    }
}
```

**Tasks:**
- [ ] Implement fused kernel for straight lines
- [ ] Implement fused kernel for Bezier curves
- [ ] Benchmark vs separate curve-gen + raster
- [ ] Expect 2-3× additional speedup

---

#### Day 7: GPU Dot Renderer
**File:** `nodes/dot_renderer.py`

**Implementation:**
```python
from ..utils.gpu_rendering import GPUDotRenderer

class DotRendererNode:
    def __init__(self):
        self.gpu_renderer = GPUDotRenderer()

    def _render_single_frame(self, ...):
        if use_gpu:
            layer = self.gpu_renderer.render_dots_batch(
                points, radii, colors, image_width, image_height
            )
        else:
            layer = self._render_dots_cpu(...)
```

**CUDA Kernel (SDF-based circles):**
```cuda
extern "C" __global__
void batched_dots_sdf(
    const float* dots,          // Nx5: [x, y, radius, r, g, b]
    float* output,
    int width, int height, int n_dots
) {
    // Similar to bbox SDF but circular distance
    // d = length(px - dot_center) - radius
}
```

**Tasks:**
- [ ] Implement GPU dot renderer with SDF
- [ ] Support variable radii per dot
- [ ] Support per-dot colors
- [ ] Benchmark: 1000 dots @ 4K should be <3ms

---

#### Day 8-9: GPU Compositing & Layer Merge
**New File:** `utils/gpu_compositing.py`

**Goal:** Merge all layers in single GPU pass (no CPU involvement)

```python
class GPUCompositor:
    """
    One-pass layer merging on GPU

    Premultiplied alpha blending in linear color space
    Supports blend modes: normal, add, screen, multiply
    """

    def merge_layers_onepass(
        self,
        layers: List[cp.ndarray],  # List of (H, W, 4) GPU arrays
        blend_modes: List[str],
        opacities: List[float]
    ) -> cp.ndarray:
        """
        Merge all layers in single GPU pass

        Read each layer once, write result once
        Much faster than sequential CPU blending
        """
        # Kernel launches once, processes all layers
        pass
```

**CUDA Kernel:**
```cuda
extern "C" __global__
void merge_layers_onepass(
    const float** layers,       // Array of layer pointers
    const int* blend_modes,     // Blend mode per layer
    const float* opacities,     // Opacity per layer
    float* output,
    int width, int height, int n_layers
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Start with transparent black
    float4 result = make_float4(0, 0, 0, 0);

    // Blend each layer
    for (int i = 0; i < n_layers; i++) {
        float4 layer_pixel = read_pixel(layers[i], x, y, width);
        layer_pixel.w *= opacities[i];  // Apply opacity

        // Blend based on mode
        result = blend(result, layer_pixel, blend_modes[i]);
    }

    write_pixel(output, x, y, width, result);
}

__device__ float4 blend(float4 dst, float4 src, int mode) {
    // Premultiplied alpha blending
    switch (mode) {
        case BLEND_NORMAL:
            return make_float4(
                src.x + dst.x * (1 - src.w),
                src.y + dst.y * (1 - src.w),
                src.z + dst.z * (1 - src.w),
                src.w + dst.w * (1 - src.w)
            );
        case BLEND_ADD:
            return make_float4(
                dst.x + src.x,
                dst.y + src.y,
                dst.z + src.z,
                src.w + dst.w * (1 - src.w)
            );
        // ... other modes
    }
}
```

**Tasks:**
- [ ] Implement one-pass merge kernel
- [ ] Support blend modes: normal, add, screen, multiply, lighten
- [ ] Integrate into CompositeOverNode
- [ ] Benchmark: merge 5 layers @ 4K should be <2ms

---

#### Day 10: FP16 Pipeline Conversion
**Files:** All rendering utils

**Goal:** Convert entire pipeline to FP16 for 2× bandwidth and 0.5× VRAM

**Changes:**
```python
# All output buffers use FP16
output = cp.zeros((height, width, 4), dtype=cp.float16)  # Was float32

# Accumulation in FP32 for precision
accum = cp.zeros((height, width, 4), dtype=cp.float32)
# ... render to accum ...
output = accum.astype(cp.float16)
```

**CUDA Kernel Updates:**
```cuda
// Change from float to half (FP16)
extern "C" __global__
void render_fp16(
    const float* points,        // Keep inputs FP32
    half* output,               // Output FP16
    // ...
) {
    // Compute in FP32
    float r, g, b, a;
    // ... rendering ...

    // Convert to FP16 on write
    output[idx] = __float2half(r);
}
```

**Tasks:**
- [ ] Convert all kernels to FP16 output
- [ ] Keep intermediate math in FP32 where needed
- [ ] Test visual quality (should be identical)
- [ ] Measure bandwidth improvement

---

### Phase 3: Optimization & Advanced Features (Week 3)

#### Day 11-12: CUDA Graphs with Persistent Buffers
**New File:** `utils/gpu_pipeline.py`

**Goal:** Eliminate kernel launch overhead, reuse buffers

```python
class GPUPipeline:
    """
    Complete rendering pipeline with CUDA Graphs

    Captures the full sequence once, replays for each frame
    Eliminates kernel launch overhead (~10-20% speedup)
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Persistent buffers (allocated once, reused)
        self.edges_buf = None
        self.line_layer = cp.zeros((height, width, 4), dtype=cp.float16)
        self.bbox_layer = cp.zeros((height, width, 4), dtype=cp.float16)
        self.dot_layer = cp.zeros((height, width, 4), dtype=cp.float16)
        self.output = cp.zeros((height, width, 4), dtype=cp.float16)

        # CUDA Graph
        self.graph = None
        self.captured = False

    def capture_graph(self, sample_points, sample_edges, sample_boxes, sample_dots):
        """Capture the full pipeline as CUDA Graph"""
        if self.captured:
            return

        # Create graph
        graph = cp.cuda.Graph()

        with graph.capture():
            # Run full pipeline once
            self._build_graph_gpu(sample_points)
            self._render_lines_gpu(sample_edges)
            self._render_boxes_gpu(sample_boxes)
            self._render_dots_gpu(sample_dots)
            self._merge_layers_gpu()

        self.graph = graph.instantiate()
        self.captured = True

    def render_frame(self, points, boxes, dots):
        """Replay CUDA Graph for new frame data"""
        # Update input buffers
        self.points_buf[:] = points
        self.boxes_buf[:] = boxes
        self.dots_buf[:] = dots

        # Replay graph (almost zero overhead)
        self.graph.launch()

        return self.output
```

**Tasks:**
- [ ] Implement CUDA Graph capture
- [ ] Allocate persistent buffers
- [ ] Measure overhead reduction
- [ ] Handle dynamic sizes gracefully

---

#### Day 13: GPU Video I/O with NVDEC/NVENC
**New File:** `utils/gpu_video.py`

**Goal:** Eliminate CPU video bottleneck

```python
class GPUVideoProcessor:
    """
    GPU-accelerated video I/O using NVDEC/NVENC

    Decode frames directly to GPU texture
    Encode output directly from GPU buffer
    Overlap decode/compute/encode on streams
    """

    def __init__(self):
        self.decoder = NVDECDecoder()
        self.encoder = NVENCEncoder()

        # Three CUDA streams for overlap
        self.stream_decode = cp.cuda.Stream()
        self.stream_compute = cp.cuda.Stream()
        self.stream_encode = cp.cuda.Stream()

    def process_video_pipelined(self, input_path, output_path, render_func):
        """
        Pipelined video processing

        Frame N:   Decode
        Frame N-1:        Render
        Frame N-2:               Encode

        Overlap adds 10-20% throughput
        """
        pass
```

**Tasks:**
- [ ] Implement NVDEC decoder wrapper
- [ ] Implement NVENC encoder wrapper
- [ ] Set up multi-stream pipeline
- [ ] Integrate with batch processing

---

#### Day 14: Advanced GPU Effects
**Files:** Extend `utils/gpu_rendering.py`

**Electric Effect (GPU Perlin Noise):**
```cuda
__global__ void electric_effect(
    // Perlin noise evaluated on GPU
    // Adds random lightning-like jitter to lines
)
```

**Particle Trail Effect:**
```cuda
__global__ void particle_trail(
    // GPU particle system along curves
    // Each curve spawns particles, simulated in parallel
)
```

**Blur/Glow Effects:**
```cuda
__global__ void separable_blur_x(/* Horizontal pass */)
__global__ void separable_blur_y(/* Vertical pass */)
// Use shared memory tiles for efficiency
```

**Tasks:**
- [ ] Implement GPU Perlin noise for electric effect
- [ ] Implement GPU particle system
- [ ] Implement separable Gaussian blur
- [ ] Add glow/bloom effects

---

### Phase 4: Testing, Optimization & Documentation (Days 15-21)

#### Day 15-16: Comprehensive Testing

**Unit Tests:**
```python
# Test each GPU component vs CPU reference
def test_gpu_bbox_vs_cpu():
    cpu = render_bbox_cpu(params)
    gpu = render_bbox_gpu(params)
    assert np.allclose(cpu, gpu, rtol=1e-3)

def test_gpu_curves_all_types():
    for curve_type in ALL_CURVE_TYPES:
        cpu = generate_curve_cpu(curve_type, params)
        gpu = generate_curve_gpu(curve_type, params)
        assert np.allclose(cpu, gpu, rtol=1e-3)
```

**Performance Benchmarks:**
```python
def benchmark_full_pipeline():
    """
    4K, 1000 points, 500 edges, 100 boxes, 1000 dots
    Target: <15ms total
    """
    points = generate_test_points(1000)

    profiler = GPUMemoryManager()
    result, time_ms, mem_mb = profiler.profile_operation(
        render_full_pipeline_gpu, points
    )

    assert time_ms < 15, f"Too slow: {time_ms}ms"
    assert mem_mb < 8000, f"Too much memory: {mem_mb}MB"
```

**Visual Regression Tests:**
```python
def test_visual_regression_full_pipeline():
    """Compare GPU output to CPU golden reference"""
    cpu_ref = load_golden_reference("test_frame.npy")
    gpu_output = render_gpu(test_params)

    # SSIM (structural similarity)
    ssim = compute_ssim(cpu_ref, gpu_output)
    assert ssim > 0.99, f"Visual quality degraded: SSIM={ssim}"
```

**Stress Tests:**
```python
def test_gpu_memory_leak():
    """Run 1000 frames, check for leaks"""
    gpu = get_gpu_accelerator()

    for i in range(1000):
        render_frame_gpu(params)

        # Check memory growth
        if i % 100 == 0:
            stats = gpu.memory_stats()
            # Should be stable, not growing
```

---

#### Day 17-18: Performance Optimization

**Profile Hot Spots:**
```bash
# Use NVIDIA Nsight Systems
nsys profile -o profile python test_rendering.py

# Analyze:
# - Kernel launch overhead
# - Memory transfer bottlenecks
# - Compute vs memory bound
```

**Optimize Based on Profile:**
- [ ] Reduce kernel launches (fuse more operations)
- [ ] Optimize memory access patterns (coalescing)
- [ ] Tune block sizes for occupancy
- [ ] Add shared memory where beneficial
- [ ] Reduce register pressure if needed

---

#### Day 19-20: Documentation

**User Documentation:**
```markdown
# GPU Acceleration Guide

## Quick Start
All nodes support GPU acceleration via `use_gpu` parameter.

## Performance Settings
- **delta_y_cap**: Prevent vertical connections (None = disabled)
- **degree_cap**: Limit connections per point (None = unlimited)
- **tile_size**: Spatial grid size (64 = default, good balance)

## Troubleshooting
- GPU not available: Install `cupy-cuda12x` and `faiss-gpu`
- Out of memory: Reduce batch size or use FP16 mode
- Slow performance: Check GPU utilization with `nvidia-smi`
```

**Developer Documentation:**
```markdown
# GPU Pipeline Architecture

## Data Flow
ComfyUI Tensor → DLPack → GPU → Render → GPU → DLPack → ComfyUI Tensor

## Adding New GPU Kernels
1. Write CUDA kernel in `utils/gpu_rendering.py`
2. Compile with `cp.RawKernel()`
3. Add CPU fallback path
4. Write unit test vs CPU reference
5. Add visual regression test
6. Benchmark performance

## Memory Management
- All large buffers in FP16
- Reuse persistent buffers with CUDA Graphs
- Clear memory pool after batch: `gpu.clear_memory()`
```

---

#### Day 21: Final Integration & Polish

**Integration Checklist:**
- [ ] All nodes have GPU paths
- [ ] CPU fallbacks work
- [ ] Parameters exposed in UI
- [ ] Error handling robust
- [ ] Logging informative
- [ ] Memory usage reasonable

**Polish:**
- [ ] Remove debug print statements
- [ ] Add progress bars for long operations
- [ ] Optimize imports (lazy load GPU libs)
- [ ] Add version checks (CUDA, CuPy, FAISS)
- [ ] Write CHANGELOG entry

---

## 📊 Success Metrics

### Performance Targets (4K Resolution)
| Operation | CPU Baseline | GPU Target | Speedup |
|-----------|-------------|-----------|---------|
| Graph Build (1000pts, k=5) | ~50ms | <2ms | 25× |
| Curve Gen (1000 curves) | ~300ms | <1ms | 300× |
| Line Render (500 edges) | ~150ms | <4ms | 37× |
| BBox Render (100 boxes) | ~200ms | <2ms | 100× |
| Dot Render (1000 dots) | ~100ms | <3ms | 33× |
| Compositing (5 layers) | ~50ms | <2ms | 25× |
| **Total Pipeline** | **~850ms** | **<14ms** | **60×** |

### Quality Targets
- [ ] Visual output matches CPU (SSIM > 0.99)
- [ ] Anti-aliasing quality maintained or improved
- [ ] No visual artifacts from FP16 precision
- [ ] Smooth animations (no frame drops)

### Memory Targets
- [ ] 4K frame: <500MB VRAM (with FP16)
- [ ] Batch of 30 frames: <8GB VRAM
- [ ] No memory leaks over 1000 frames

### Code Quality Targets
- [ ] Unit test coverage: >80%
- [ ] All GPU paths have CPU fallbacks
- [ ] Visual regression tests passing
- [ ] Performance benchmarks documented
- [ ] API documentation complete

---

## 🛠️ Dependencies & Installation

### Required
```bash
# GPU Libraries
pip install cupy-cuda12x==12.3.0      # CuPy for GPU operations
pip install faiss-gpu                 # FAISS for fast KNN
pip install torch torchvision         # PyTorch for GPU utils

# Scientific Computing
pip install numpy scipy scikit-image
pip install opencv-python
```

### Optional (Advanced Features)
```bash
# RAPIDS (alternative to FAISS)
pip install cuml-cu12

# NVIDIA Video Processing
pip install nvidia-vpi

# Profiling
pip install nvtx
```

### System Requirements
- NVIDIA GPU with CUDA 12.x support
- RTX 5090 recommended (24GB VRAM)
- CUDA Toolkit 12.x installed
- NVIDIA Driver ≥ 525.x

---

## 🚨 Risk Management

### Risk: GPU Not Available
**Mitigation:** CPU fallback always maintained
**Detection:** Check `is_gpu_available()` at node init
**Recovery:** Graceful degradation with warning message

### Risk: GPU Memory Overflow
**Mitigation:**
- Batch size limits based on available VRAM
- FP16 pipeline reduces memory 2×
- Persistent buffers with CUDA Graphs
**Detection:** Monitor memory usage per frame
**Recovery:** Reduce batch size, process in chunks

### Risk: Visual Quality Regression
**Mitigation:**
- Automated visual regression tests
- SSIM threshold enforcement
- Manual review of edge cases
**Detection:** CI/CD visual diff checks
**Recovery:** Adjust AA parameters, increase precision where needed

### Risk: Performance Worse Than Expected
**Mitigation:**
- Profile-guided optimization
- Incremental approach (can roll back)
- Fallback to previous implementation
**Detection:** Automated benchmarks in CI/CD
**Recovery:** Optimize hot spots, consider hybrid CPU/GPU

---

## 📈 Rollout Plan

### Alpha (Week 1-2)
- GPU bbox and line renderers
- Limited testing
- Internal use only

### Beta (Week 3)
- Full GPU pipeline
- Comprehensive testing
- Early adopter feedback

### Production (Week 4)
- All tests passing
- Documentation complete
- Release to users

---

## 🎯 Next Immediate Steps

**Day 1 Morning:**
1. [ ] Integrate GPU bbox renderer into `BBoxRenderer` node
2. [ ] Add performance logging
3. [ ] Run initial benchmark

**Day 1 Afternoon:**
4. [ ] Write unit test for GPU bbox vs CPU
5. [ ] Write visual regression test
6. [ ] Document new `use_gpu` parameter

**Ready to start when you are! 🚀**
