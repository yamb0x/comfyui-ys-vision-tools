# YS-vision-tools System Architecture

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools for ComfyUI
**Developer:** Yambo Studio

---

## Overview

YS-vision-tools is a ComfyUI custom node package implementing GPU-accelerated vision processing. This document describes the core architectural patterns that all nodes follow.

**Key Design Principles:**
- GPU-first with CPU fallback
- Batch processing for video
- Temporal state persistence
- Standardized data types
- Performance logging

---

## Node Architecture

### Node Structure

All nodes follow a standardized ComfyUI pattern:

```python
class YourNode:
    """
    Display name, purpose, and usage description
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters with types and constraints"""
        return {
            "required": {
                "param_name": ("TYPE", {"default": value, "min": 0, "max": 100}),
            },
            "optional": {
                "optional_param": ("TYPE",),
            }
        }

    RETURN_TYPES = ("TYPE1", "TYPE2")  # Output types
    RETURN_NAMES = ("output1", "output2")  # Output labels (optional)
    FUNCTION = "execute"  # Method name to call
    CATEGORY = "YS-vision-tools/Category"  # Menu location

    def execute(self, **kwargs):
        """
        Main execution method

        Returns: Tuple matching RETURN_TYPES
        """
        # Process inputs
        result = self._process(**kwargs)
        return (result,)
```

### Node Registration

Nodes are registered in `__init__.py`:

```python
# Import node classes
from .nodes.your_node import YourNodeClass

# Register with ComfyUI
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS["YS_NodeID"] = YourNodeClass
NODE_DISPLAY_NAME_MAPPINGS["YS_NodeID"] = "Display Name 🎯"
```

**Registration Pattern:**
- Node ID: `YS_NodeName` (internal identifier)
- Display Name: `"Readable Name 🎯"` (shown in UI)
- Emoji: Used for visual identification in node menu

### Project Structure

```
custom_nodes/ys_vision_tools/
├── __init__.py               # Node registration
├── nodes/                    # All node implementations
│   ├── track_detect.py       # Object/Motion tracker
│   ├── track_detect_v2.py    # Colors/Luma tracker
│   ├── bbox_renderer.py      # GPU-accelerated bbox
│   ├── echo_layer.py         # Temporal effects
│   └── ... (18 more nodes)
├── utils/                    # Shared utilities
│   ├── gpu_common.py         # GPU acceleration helpers
│   ├── gpu_rendering.py      # GPU render utilities
│   ├── curve_math.py         # Mathematical curves
│   ├── image_utils.py        # Format conversions
│   ├── color_utils.py        # Color parsing
│   ├── sdf_font.py           # Text rendering
│   └── cuda_kernels.py       # Custom CUDA kernels
└── data/                     # Font and preset data
    └── fonts/                # System font cache
```

---

## Data Type System

### Core Types

**TRACKS** - Point coordinates for tracking
```python
# Format: NumPy array (N, 2) where N is number of points
tracks = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)

# For video: List of track arrays (one per frame)
batch_tracks = [
    np.array([[x, y], ...]),  # Frame 0
    np.array([[x, y], ...]),  # Frame 1
    ...
]
```

**STATE** - Internal temporal state for video
```python
# Format: Dictionary with persistent data across frames
state = {
    "accumulator": np.array(...),  # Temporal accumulation
    "track_ids": dict(),           # Track ID mapping
    "frame_count": int,            # Frame counter
    # ... node-specific state
}

# Pass between frames for temporal effects
output, new_state = node.execute(input, prev_state=state)
```

**LAYER** - RGBA overlay with transparency
```python
# Format: PyTorch tensor (B, H, W, 4) where 4 = RGBA
# B = batch size (number of frames)
# H, W = image dimensions
# Values: 0.0 to 1.0 (float32)

layer = torch.zeros((1, 2160, 3840, 4), dtype=torch.float32)
# [:, :, :, 0:3] = RGB channels
# [:, :, :, 3] = Alpha channel (transparency)
```

**IMAGE** - RGB image without transparency
```python
# Format: PyTorch tensor (B, H, W, 3) where 3 = RGB
# ComfyUI uses BHWC format (not BCHW)
image = torch.zeros((1, 2160, 3840, 3), dtype=torch.float32)
```

**BOXES** - Bounding box coordinates
```python
# Format: NumPy array (N, 4) where 4 = [x, y, width, height]
boxes = np.array([[x, y, w, h], ...], dtype=np.float32)
```

**AGES** - Temporal age tracking
```python
# Format: NumPy array (N,) with age per point/box
ages = np.array([age1, age2, ...], dtype=np.float32)
# Used for fade effects, temporal sizing, etc.
```

### Type Conversions

**ComfyUI ↔ NumPy**
```python
# ComfyUI tensor to NumPy (BHWC format)
def comfyui_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI image tensor to numpy array"""
    return tensor.cpu().numpy()  # Already BHWC

# NumPy to ComfyUI tensor
def numpy_to_comfyui(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to ComfyUI image tensor"""
    return torch.from_numpy(array).float()  # Keep BHWC
```

**GPU Transfers**
```python
# CPU → GPU
gpu_array = cp.asarray(cpu_array)

# GPU → CPU
cpu_array = cp.asnumpy(gpu_array)
```

---

## Batch Processing Architecture

### Video as Batched Tensors

ComfyUI passes video as batched tensors with shape `(B, H, W, C)`:
- B = number of frames (batch size)
- H, W = image dimensions
- C = channels (3 for IMAGE, 4 for LAYER)

**CRITICAL:** All nodes MUST process ALL frames in the batch.

### Standard Batch Processing Pattern

```python
def execute(self, layer: torch.Tensor, param: float, prev_state=None):
    """
    Process all frames in batch

    Args:
        layer: Input tensor (B, H, W, C)
        param: Node parameter
        prev_state: Previous frame state (for temporal nodes)
    """
    # 1. Detect batch size
    batch_size = layer.shape[0]
    is_batch = batch_size > 1

    if is_batch:
        print(f"[YS-NODE] BATCH MODE: {batch_size} frames")

    # 2. Initialize state (OUTSIDE loop for temporal nodes)
    if prev_state is None:
        state = self._init_state(layer.shape)
    else:
        state = prev_state

    # 3. Process EACH frame
    output_frames = []
    for i in range(batch_size):  # CRITICAL: range(batch_size), not [0]
        # Get current frame
        frame_np = layer[i].cpu().numpy()

        # Process frame (state persists for temporal effects!)
        result_np, state = self._process_frame(frame_np, state, param)

        output_frames.append(result_np)

        # Progress logging (every 10 frames to avoid spam)
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

### Temporal State Management

For nodes with temporal effects (Echo Layer, trails, accumulation):

**State Initialization (once per video)**
```python
def _init_state(self, shape):
    """Initialize state for first frame"""
    B, H, W, C = shape
    return {
        "accumulator": np.zeros((H, W, C), dtype=np.float32),
        "frame_count": 0,
        # ... other state
    }
```

**State Updates (every frame)**
```python
def _process_frame(self, frame, state, param):
    """Update state for current frame"""
    # Use previous state
    accumulator = state["accumulator"]

    # Update with current frame
    accumulator = accumulator * 0.9 + frame * 0.1

    # Store updated state
    state["accumulator"] = accumulator
    state["frame_count"] += 1

    return result, state
```

**State Loop in Workflow**
```
Echo Layer
  ↓ output
  ↓ STATE → (loop back to STATE input)
  ↓
Next Node
```

---

## GPU/CPU Fallback Pattern

### Standard Implementation

All computationally intensive operations MUST provide GPU acceleration:

```python
def process(data: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Standard GPU/CPU pattern

    Args:
        data: Input array
        use_gpu: Enable GPU acceleration (default True)

    Returns:
        Processed array
    """
    if use_gpu and CUPY_AVAILABLE:
        # GPU path
        start_time = time.perf_counter()

        # Transfer to GPU
        data_gpu = cp.asarray(data)

        # Process on GPU
        result_gpu = gpu_operation(data_gpu)

        # Transfer back
        result = cp.asnumpy(result_gpu)

        gpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-NODE] GPU operation in {gpu_time:.2f}ms")

        return result

    else:
        # CPU fallback
        if use_gpu and not CUPY_AVAILABLE:
            print("[YS-NODE] GPU requested but CuPy unavailable, using CPU")

        start_time = time.perf_counter()
        result = cpu_operation(data)
        cpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-NODE] CPU operation in {cpu_time:.2f}ms")

        return result
```

### GPU Availability Detection

```python
# In utils/gpu_common.py
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[YS-VISION] CuPy available - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("[YS-VISION] CuPy not available - CPU fallback only")
```

### Custom CUDA Kernels

For maximum performance, critical operations use custom CUDA kernels:

```python
# Example: Echo Layer EMA update kernel
cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void ema_update(
    const float* input,
    float* accumulator,
    const float decay,
    const int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        accumulator[idx] = accumulator[idx] * decay + input[idx] * (1.0f - decay);
    }
}
''', 'ema_update')

# Launch kernel
threads_per_block = 256
blocks = (size + threads_per_block - 1) // threads_per_block
cuda_kernel((blocks,), (threads_per_block,), (input_gpu, accum_gpu, decay, size))
```

**Custom Kernels in Project:**
1. **Echo Layer EMA** - Exponential moving average update (3-5× speedup)
2. **BBox SDF Rendering** - Signed distance field rendering (2000× speedup)
3. **Track Masking** - Binary mask generation for track regions

### Memory Management

```python
# Limit GPU memory usage (safe for 24GB VRAM)
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=8 * 1024**3)  # 8GB max per operation

# Allow PyTorch to use remaining memory
torch.cuda.set_per_process_memory_fraction(0.8)  # 80% for PyTorch
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat32
```

---

## Performance Conventions

### Logging Format

```python
# GPU timing
print(f"[YS-NODENAME] GPU operation @ {W}x{H} in {time_ms:.2f}ms")

# CPU timing
print(f"[YS-NODENAME] CPU operation @ {W}x{H} in {time_ms:.2f}ms")

# Batch processing
print(f"[YS-NODENAME] BATCH MODE: {batch_size} frames")
print(f"[YS-NODENAME] Processed frame {i+1}/{batch_size}")

# GPU availability
print(f"[YS-NODENAME] GPU requested but unavailable, using CPU")
```

### Benchmarking Pattern

```python
def benchmark_operation(iterations=100):
    """Benchmark GPU vs CPU performance"""
    # Warmup
    for _ in range(5):
        gpu_operation(test_data)

    # Time GPU
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_operation(test_data)
    gpu_time = (time.perf_counter() - start) / iterations

    # Time CPU
    start = time.perf_counter()
    for _ in range(iterations):
        cpu_operation(test_data)
    cpu_time = (time.perf_counter() - start) / iterations

    speedup = cpu_time / gpu_time
    print(f"GPU: {gpu_time*1000:.2f}ms | CPU: {cpu_time*1000:.2f}ms | Speedup: {speedup:.1f}×")
```

---

## Error Handling

### Input Validation

```python
def execute(self, tracks, image, param):
    # Validate tracks format
    if isinstance(tracks, list):
        # Batch mode
        if len(tracks) == 0:
            raise ValueError("Empty track batch")
    else:
        # Single frame
        if tracks.shape[1] != 2:
            raise ValueError(f"Expected tracks shape (N, 2), got {tracks.shape}")

    # Validate parameter ranges
    if not 0.0 <= param <= 1.0:
        raise ValueError(f"Parameter must be in [0, 1], got {param}")

    # Process...
```

### GPU Error Handling

```python
try:
    result = gpu_operation(data_gpu)
except cp.cuda.memory.OutOfMemoryError:
    print("[YS-NODE] GPU out of memory, falling back to CPU")
    result = cpu_operation(data_cpu)
except Exception as e:
    print(f"[YS-NODE] GPU error: {e}, falling back to CPU")
    result = cpu_operation(data_cpu)
```

---

## Testing Approach

### ⚠️ CRITICAL: Always Test in ComfyUI

**DO NOT use pytest or standalone Python tests.**

Proper testing workflow:
1. Make code changes on D: drive (source of truth)
2. Copy to F: drive ComfyUI installation
3. Restart ComfyUI server
4. Create test workflow with the node
5. Run and observe console output
6. Verify visual results and performance logs

### Test Cases

**Edge Cases to Test:**
- Empty input (no tracks/points)
- Single point
- Large batch (1000+ points)
- 4K resolution (3840×2160)
- Video batch (50+ frames)
- GPU disabled (`use_gpu=False`)

**Visual Verification:**
- Output looks correct
- No artifacts or glitches
- Anti-aliasing quality good
- Colors match expected
- Transparency/alpha blending correct

**Performance Verification:**
- Check console logs for `[YS-XXX] GPU rendered...`
- Verify GPU/CPU fallback works
- Confirm batch processing logs appear for video

---

## References

- **Main README:** `README.md` - User guide and node overview
- **Node Catalog:** `docs/NODE_CATALOG.md` - Complete node reference
- **GPU Acceleration:** `docs/GPU_ACCELERATION.md` - GPU implementation details
- **Development Rules:** `CLAUDE.md` - Project-specific guidelines

---

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools
**Developer:** Yambo Studio
