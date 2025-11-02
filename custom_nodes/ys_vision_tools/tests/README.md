# YS-vision-tools Test Suite

## Quick Start

### Option 1: Minimal Standalone Test (Recommended First)
```bash
cd F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\tests
python test_minimal.py
```

This will:
- Test imports
- Check GPU availability
- Test CPU rendering
- Test GPU rendering (if available)
- Compare GPU vs CPU output

### Option 2: Full Test Suite (Requires pytest)

1. **Install test dependencies:**
```bash
pip install opencv-python pytest
```

2. **Run tests:**
```bash
# All GPU bbox tests
pytest test_gpu_bbox_renderer.py -v

# Visual regression tests
pytest test_gpu_visual_regression.py -v -m visual

# Or use the batch script
run_tests.bat
```

## Missing Dependencies?

If you see `ModuleNotFoundError: No module named 'cv2'`:
```bash
pip install opencv-python
```

If you want GPU acceleration:
```bash
# For RTX 5090 (CUDA 12.x)
pip install cupy-cuda12x
pip install faiss-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Test Output

### Minimal Test Output
```
==============================================================
YS-vision-tools GPU BBox Renderer - Minimal Test
==============================================================

[1/5] Testing imports...
✓ Imports successful

[2/5] Checking GPU availability...
  GPU Available: True

[3/5] Initializing BBoxRenderer node...
✓ Node initialized
✓ GPU renderer initialized

[4/5] Testing CPU rendering...
[YS-BBOX] CPU rendered 3 boxes @ 1920x1080 in 12.34ms
✓ CPU rendering successful: 12.34ms
  Output shape: torch.Size([1, 1080, 1920, 4])

[5/5] Testing GPU rendering...
[YS-BBOX] GPU rendered 3 boxes @ 1920x1080 in 1.23ms
✓ GPU rendering successful: 1.23ms
  Output shape: torch.Size([1, 1080, 1920, 4])
✓ GPU output matches CPU (within tolerance)
  Speedup: 10.0×

==============================================================
✓ All tests passed!
==============================================================
```

### Full Test Suite Output
```
test_gpu_bbox_renderer.py::TestGPUBBoxRenderer::test_gpu_available PASSED
test_gpu_bbox_renderer.py::TestGPUBBoxRenderer::test_single_box_gpu_vs_cpu PASSED
test_gpu_bbox_renderer.py::TestGPUBBoxRenderer::test_multiple_boxes_gpu_vs_cpu PASSED
test_gpu_bbox_renderer.py::TestGPUBBoxRenderer::test_gpu_performance_4k PASSED

4K Benchmark (100 boxes):
  CPU: 187.56ms
  GPU: 2.34ms
  Speedup: 80.2×
```

## Visual Regression Tests

Visual tests save comparison images to `visual_output/`:
- `single_box.png` - Single orange box
- `grid_boxes.png` - 5×5 grid of cyan boxes
- `roundness_square.png` - Square corners
- `roundness_semi.png` - Semi-rounded corners
- `roundness_circle.png` - Circular (roundness=1.0)
- `stress_200_boxes.png` - Stress test with 200 boxes

Each image shows: [CPU | GPU | Diff (10×)]

## Troubleshooting

### Import Errors
If you see import errors, make sure you're running from the tests directory:
```bash
cd F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\tests
python test_minimal.py
```

### GPU Not Available
The tests will automatically fall back to CPU if GPU is not available.
You'll see warnings like:
```
GPU libraries not available. Install cupy-cuda12x and torch for GPU acceleration.
```

### Package Structure Issues
The tests try multiple import strategies:
1. Direct import from `custom_nodes.ys_vision_tools`
2. Fallback to `ys_vision_tools`

If both fail, check that you're running from the correct directory.

## Performance Benchmarks

Expected performance on RTX 5090:

| Test | Resolution | Boxes | CPU | GPU | Speedup |
|------|-----------|-------|-----|-----|---------|
| Light | 1080p | 10 | ~5ms | ~0.5ms | 10× |
| Medium | 1080p | 100 | ~50ms | ~2ms | 25× |
| Heavy | 4K | 100 | ~200ms | ~2ms | 100× |
| Stress | 4K | 1000 | ~2000ms | ~20ms | 100× |

## Next Steps

After Day 1 tests pass:
1. Test in ComfyUI workflow
2. Verify animation batching (50 frames)
3. Move to Day 2: GPU Graph Builder integration
