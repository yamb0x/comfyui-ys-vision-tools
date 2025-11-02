# Common Pitfalls & Troubleshooting Guide

## 🚨 Most Common Mistakes (and How to Avoid Them)

### 1. ❌ Wrong Image Format (NumPy vs Torch vs OpenCV)

**The Problem:**
```python
# This will crash!
image = torch.rand(3, 480, 640)  # Torch format: (C, H, W)
cv2.GaussianBlur(image, (5, 5), 1.0)  # OpenCV expects NumPy (H, W, C)
```

**The Solution:**
```python
# Always convert formats explicitly
def ensure_numpy_hwc(image):
    """Convert any image format to NumPy (H, W, C)"""
    if isinstance(image, torch.Tensor):
        # Torch tensor -> NumPy
        image = image.cpu().numpy()

        # Handle different dimension orders
        if image.ndim == 4:  # (B, C, H, W)
            image = image[0].transpose(1, 2, 0)
        elif image.ndim == 3:
            if image.shape[0] in [1, 3, 4]:  # (C, H, W)
                image = image.transpose(1, 2, 0)
            # else already (H, W, C)

    return image

# Now safe to use with OpenCV
image = ensure_numpy_hwc(image)
result = cv2.GaussianBlur(image, (5, 5), 1.0)
```

### 2. ❌ Float vs Uint8 Confusion

**The Problem:**
```python
# ComfyUI uses float [0, 1], OpenCV often expects uint8 [0, 255]
image = np.random.rand(480, 640, 3)  # Float [0, 1]
edges = cv2.Canny(image, 100, 200)  # WRONG! Expects uint8
```

**The Solution:**
```python
# Convert when needed, but track format
def to_uint8(image):
    """Convert float [0,1] to uint8 [0,255]"""
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

def to_float(image):
    """Convert uint8 [0,255] to float [0,1]"""
    return image.astype(np.float32) / 255.0

# Use OpenCV function
image_uint8 = to_uint8(image)
edges = cv2.Canny(image_uint8, 100, 200)
edges_float = to_float(edges)  # Convert back for ComfyUI
```

### 3. ❌ Coordinate System Confusion

**The Problem:**
```python
# Is it (x, y) or (y, x)? Row-major or column-major?
point = [100, 200]
image[point[0], point[1]] = 1  # WRONG! Arrays are [row, col] = [y, x]
```

**The Solution:**
```python
# Be explicit about coordinate systems
class Point:
    def __init__(self, x, y):
        self.x = x  # Column (width dimension)
        self.y = y  # Row (height dimension)

    def to_array_index(self):
        """Convert to array indexing [row, col]"""
        return (self.y, self.x)

    def to_xy(self):
        """Get as (x, y) tuple"""
        return (self.x, self.y)

# Use consistently
point = Point(x=100, y=200)
image[point.to_array_index()] = 1  # Correct!
cv2.circle(image, point.to_xy(), radius=5, color=1)  # Also correct!
```

### 4. ❌ Memory Leaks with Large Arrays

**The Problem:**
```python
class BadNode:
    def __init__(self):
        self.cache = []  # Keeps growing!

    def execute(self, image):
        processed = expensive_operation(image)
        self.cache.append(processed)  # Memory leak!
        return processed
```

**The Solution:**
```python
class GoodNode:
    def __init__(self):
        self.cache = None  # Single item, not list

    def execute(self, image):
        processed = expensive_operation(image)

        # Only keep latest, release old
        self.cache = processed.copy()

        # Or use bounded cache
        if hasattr(self, 'cache_list'):
            self.cache_list.append(processed)
            if len(self.cache_list) > 10:
                self.cache_list.pop(0)  # Remove oldest

        return processed
```

### 5. ❌ Premultiplied Alpha Mistakes

**The Problem:**
```python
# Incorrect alpha blending
layer = np.array([1, 0, 0, 0.5])  # Red with 50% alpha
result = image + layer[:3]  # WRONG! Ignores alpha
```

**The Solution:**
```python
def blend_with_alpha(base, overlay, premultiplied=False):
    """Correctly blend RGBA overlay onto RGB base"""

    if not premultiplied:
        # Premultiply RGB by alpha
        overlay_rgb = overlay[:, :, :3] * overlay[:, :, 3:4]
    else:
        overlay_rgb = overlay[:, :, :3]

    alpha = overlay[:, :, 3:4]

    # Correct alpha blending formula
    result = base * (1 - alpha) + overlay_rgb

    return result
```

## 🐛 ComfyUI-Specific Issues

### Issue: Node Doesn't Appear in ComfyUI

**Diagnosis Checklist:**
```python
# 1. Check __init__.py has correct mappings
NODE_CLASS_MAPPINGS = {
    "YSVisionTrackDetect": TrackDetectNode,  # Must match class name
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSVisionTrackDetect": "Track Detect"  # Display name in UI
}

# 2. Check class has required class methods
class TrackDetectNode:
    @classmethod  # <-- Don't forget this!
    def INPUT_TYPES(cls):
        return {...}

    RETURN_TYPES = (...)  # Required
    FUNCTION = "execute"  # Required
    CATEGORY = "YS-Vision"  # For menu organization

# 3. Check for import errors
# Run: python -c "from custom_nodes.ys_vision import *"
```

### Issue: Tensor Shape Mismatch

**Common ComfyUI Tensor Shapes:**
```python
# ComfyUI IMAGE format
# Shape: (batch, height, width, channels)
# Range: [0, 1] float32

def ensure_comfyui_format(image):
    """Convert any image to ComfyUI format"""

    # Ensure 4D
    if image.ndim == 2:  # Grayscale
        image = image[:, :, np.newaxis]  # Add channel
        image = np.repeat(image, 3, axis=2)  # Convert to RGB

    if image.ndim == 3:  # Single image
        image = image[np.newaxis, ...]  # Add batch

    # Ensure float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Should now be (B, H, W, C) with float [0, 1]
    assert image.ndim == 4
    assert image.dtype == np.float32
    assert 0 <= image.min() <= image.max() <= 1

    return image
```

### Issue: Node Execution Order Problems

**The Problem:** Nodes execute in wrong order, causing failures

**The Solution:**
```python
class DependentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("TRACKS",),  # This creates dependency
                "palette": ("PALETTE",)  # This too
            }
        }

    # ComfyUI automatically ensures inputs are ready before execution
    # Don't try to access other nodes directly!

    def execute(self, tracks, palette):
        # tracks and palette are guaranteed to be computed
        return process(tracks, palette)
```

## 🔍 Debugging Techniques

### 1. Add Debug Logging

Create `debug_utils.py`:
```python
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

def debug_tensor(tensor, name="tensor"):
    """Log tensor info for debugging"""
    logger = logging.getLogger("YS-Vision")

    logger.debug(f"{name} shape: {tensor.shape}")
    logger.debug(f"{name} dtype: {tensor.dtype}")
    logger.debug(f"{name} range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    if tensor.ndim == 4:
        logger.debug(f"{name} format: (batch={tensor.shape[0]}, "
                    f"h={tensor.shape[1]}, w={tensor.shape[2]}, "
                    f"c={tensor.shape[3]})")

    return tensor  # Pass through for chaining
```

### 2. Visual Debugging

```python
def save_debug_image(array, filename="debug.png"):
    """Save array as image for visual debugging"""
    from PIL import Image

    # Handle different formats
    if array.ndim == 4:
        array = array[0]  # Take first batch
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)  # Gray to RGB

    # Ensure valid range
    if array.dtype == np.float32:
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)

    Image.fromarray(array).save(filename)
    print(f"Debug image saved: {filename}")
```

### 3. Performance Profiling

```python
import time
import functools

def profile_time(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper

# Usage
@profile_time
def slow_function():
    time.sleep(0.1)
    return "done"
```

## 🔧 OpenCV Gotchas

### Drawing Functions Modify In-Place
```python
# WRONG - modifies original!
image = load_image()
cv2.circle(image, (100, 100), 10, (255, 0, 0), -1)
return image  # Original is changed!

# RIGHT - work on copy
image = load_image()
result = image.copy()
cv2.circle(result, (100, 100), 10, (255, 0, 0), -1)
return result  # Original unchanged
```

### Coordinate Types Matter
```python
# WRONG - float coordinates
cv2.line(image, (50.5, 100.7), (150.3, 200.9), color)

# RIGHT - convert to int
pt1 = tuple(map(int, (50.5, 100.7)))
pt2 = tuple(map(int, (150.3, 200.9)))
cv2.line(image, pt1, pt2, color)
```

## 💾 Memory Management Tips

### 1. Release Large Arrays
```python
def process_video(frames):
    results = []

    for frame in frames:
        # Process frame
        large_temp = compute_something_big(frame)
        result = extract_small_result(large_temp)

        # Explicitly release memory
        del large_temp

        results.append(result)

    # Force garbage collection for large batches
    if len(frames) > 100:
        import gc
        gc.collect()

    return results
```

### 2. Use Views Instead of Copies
```python
# Inefficient - creates copy
roi = image[100:200, 100:200].copy()
process(roi)

# Efficient - creates view (if read-only)
roi = image[100:200, 100:200]
result = read_only_process(roi)
```

### 3. Preallocate Arrays
```python
# Inefficient
results = []
for i in range(1000):
    results.append(compute(i))
results = np.array(results)

# Efficient - preallocate
results = np.zeros((1000, 100, 100, 4))
for i in range(1000):
    results[i] = compute(i)
```

## 🎯 Quick Fixes for Common Errors

### "ValueError: too many values to unpack"
```python
# Problem
x, y = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fix - OpenCV version compatibility
contours, hierarchy = cv2.findContours(...)[-2:]  # Works for any version
```

### "TypeError: Required argument 'mat' (pos 1) not found"
```python
# Problem
cv2.imshow(image)  # Missing window name

# Fix
cv2.imshow("Window Name", image)
```

### "AttributeError: 'NoneType' object has no attribute 'shape'"
```python
# Problem
image = cv2.imread("missing_file.jpg")
print(image.shape)  # Crashes - image is None

# Fix
image = cv2.imread("file.jpg")
if image is None:
    raise FileNotFoundError("Could not load image")
print(image.shape)
```

## 📚 Learning Resources

### For Computer Vision
- OpenCV Documentation: https://docs.opencv.org/
- PyImageSearch Tutorials: Great for practical CV

### For NumPy Array Manipulation
- NumPy User Guide: https://numpy.org/doc/stable/user/
- Visual NumPy: https://jalammar.github.io/visual-numpy/

### For ComfyUI Development
- ComfyUI Examples Repo
- Discord #dev-chat channel
- Study existing custom nodes

## 🚀 Performance Optimization Checklist

Before declaring your node "done":

- [ ] Profile with large images (4K)
- [ ] Test with batch processing
- [ ] Check memory usage over time
- [ ] Optimize hot loops with NumPy vectorization
- [ ] Consider caching expensive computations
- [ ] Use appropriate data types (float32 vs float64)
- [ ] Minimize array copies
- [ ] Release temporary arrays explicitly
- [ ] Consider GPU acceleration for heavy ops

## Next Steps
Continue to `06-PROJECT-SUMMARY.md` for final overview and checklist.