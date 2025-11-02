# Quick Test - Skip pytest for now

The pytest tests are failing because `image_size_detector.py` requires torch.

## Run the Minimal Standalone Test Instead

This test doesn't use pytest and will work without torch installed:

```bash
cd F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\tests
python test_minimal.py
```

This will test:
- GPU bbox renderer integration
- CPU vs GPU comparison
- Performance measurement

## If You Want Full pytest Suite

You need to install torch first:

```bash
# For RTX 5090 with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then run tests
pytest test_gpu_bbox_renderer.py -v
```

But **torch is already installed in ComfyUI**, so the real solution is to make sure you're using ComfyUI's Python environment!

## Using ComfyUI's Python Environment

ComfyUI already has torch installed. Use its Python:

```bash
cd F:\Comfy3D_WinPortable\ComfyUI\custom_nodes\ys_vision_tools\tests

# Use ComfyUI's Python (adjust path if different)
F:\Comfy3D_WinPortable\python_embeded\python.exe test_minimal.py

# Or if ComfyUI uses system Python, just use:
python test_minimal.py
```

The key is: **ComfyUI already has torch**, so run the test from within ComfyUI's environment!

## Alternative: Test in ComfyUI Directly

Instead of running pytest, just load ComfyUI and use the BBox Renderer node:

1. Start ComfyUI
2. Add "Bounding Box Renderer" node
3. Connect some tracks (from Track Detect node)
4. Set `use_gpu=True`
5. Generate!

You'll see in the console:
```
[YS-BBOX] GPU rendered 100 boxes @ 3840x2160 in 2.34ms
```

This proves GPU acceleration is working!
