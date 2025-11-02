# YS-vision-tools Troubleshooting Guide

## ❌ Error: "Cannot handle this data type: (1, 1, 2176), |u1"

### Root Cause (FIXED in latest version)
This error was caused by incorrect tensor format conversion. ComfyUI uses **BHWC** (Batch, Height, Width, Channels) format, NOT the standard PyTorch **BCHW** format.

The `numpy_to_comfyui()` function in `utils/image_utils.py` was incorrectly transposing tensors to BCHW format, causing dimension corruption.

### The Fix
**✅ FIXED:** Updated `numpy_to_comfyui()` function to maintain BHWC format:
```python
# WRONG (old code):
image = np.transpose(image, (2, 0, 1))  # HWC → CHW
image = image[np.newaxis, ...]          # (1, C, H, W) ❌

# CORRECT (new code):
image = image[np.newaxis, ...]          # (1, H, W, C) ✅
```

### What to Do
1. **Ensure you have the latest code** with the fix in `utils/image_utils.py`
2. **Restart ComfyUI completely** (close and reopen)
3. **Refresh your browser** (Ctrl+F5 or Cmd+Shift+R)
4. **Try your workflow again**

### If You Still See This Error
Double-check that `image_width` and `image_height` in **Line Link Renderer** match your actual image dimensions:
- If your image is 1024x1024, set width=1024, height=1024
- If your image is 512x768, set width=512, height=768
- If your image is 1920x1080, set width=1920, height=1080

### Quick Test Workflow

Try this simple workflow first:

```
1. Load Image (load any image)
2. Track Detect
   - Method: gradient_magnitude
   - Sensitivity: 0.5
   - Points: 100  ← Start with fewer points

3. Line Link Renderer
   - image_width: MATCH YOUR IMAGE!  ← Important!
   - image_height: MATCH YOUR IMAGE!
   - Curve: straight
   - Style: solid

4. Composite Over
   - opacity: 0.8

5. Preview Image
```

### Alternative: Use Debug Visualization

If you're still having issues, try viewing the debug output from Track Detect first:

```
1. Load Image
2. Track Detect
3. Preview Image ← Connect to debug_viz output
```

This will show you what points are being detected without any rendering errors.

---

## ⚠️ Common Issues

### Issue 1: No points detected
**Symptom:** Black/empty output
**Solution:**
- Increase sensitivity (try 0.7-0.9)
- Try different detection methods (gradient_magnitude is most reliable)
- Ensure image has features (edges, corners, contrast)

### Issue 2: Too many points / slow performance
**Symptom:** Very slow, excessive lines
**Solution:**
- Reduce `points_per_frame` (try 50-100 first)
- Reduce `k_neighbors` (try 2-3)
- Use simpler curve types ("straight" is fastest)

### Issue 3: GPU warnings
**Symptom:** "GPU not available" warning
**Solution:**
- This is OK! Nodes work on CPU
- To enable GPU: `pip install cupy-cuda12x torch`
- GPU is optional, not required

### Issue 4: YOLO detection doesn't work
**Symptom:** Error with "object_detection" method
**Solution:**
- Install YOLO: `pip install ultralytics`
- Or use other detection methods (they work without YOLO)

---

## 🔧 Restart After Installing Fix

After I've updated the code:

1. **Close ComfyUI** completely
2. **Restart ComfyUI**
3. **Refresh your browser** (Ctrl+F5 or Cmd+Shift+R)
4. **Try the simple test workflow** above

---

## 📊 Verify Installation

Check that nodes loaded correctly:

Look in ComfyUI console for:
```
ys_vision_tools: 2.6 seconds
```

Should see 6 nodes in menu:
- Track Detect (Enhanced) 🎯
- Line Link Renderer (Advanced) 🌀
- Dot Renderer ⚫
- Palette Map 🎨
- Layer Merge 🔀
- Composite Over 🎬

---

## 🐛 Still Having Issues?

1. **Check console** for specific error messages
2. **Try CPU mode** (set use_gpu=False)
3. **Start simple**: straight lines, solid style, 100 points
4. **Match dimensions**: This is the #1 cause of errors!

---

## ✅ Working Example Settings

This is guaranteed to work with a 512x512 image:

**Track Detect:**
- Method: gradient_magnitude
- Sensitivity: 0.6
- Points: 100
- use_gpu: True (or False if no GPU)

**Line Link Renderer:**
- Width: 512  ← Match your image!
- Height: 512  ← Match your image!
- Curve: straight
- Style: solid
- Width px: 2.0
- k_neighbors: 3

**Composite Over:**
- Opacity: 0.8
- resize_to_base: True

---

**Last Updated:** November 2, 2025 - Fixed BHWC tensor format issue
**Status:** ✅ Root cause identified and fixed! Restart ComfyUI to apply.
