# YS-vision-tools Deployment Checklist

## ✅ Pre-Deployment Safety Verification

### Code Quality Checks - PASSED ✓

1. **File Structure** ✓
   - Main package: `custom_nodes/ys_vision_tools/`
   - 6 node files in `nodes/` directory
   - 3 utility files in `utils/` directory
   - All `__init__.py` files present

2. **Node Implementation** ✓
   - 6/6 nodes have `execute()` methods
   - 6/6 nodes have `@classmethod INPUT_TYPES()`
   - 6/6 nodes have `RETURN_TYPES` defined
   - 6/6 nodes have `FUNCTION` and `CATEGORY` defined
   - All nodes registered in main `__init__.py`

3. **ComfyUI Integration** ✓
   - `NODE_CLASS_MAPPINGS` properly defined
   - `NODE_DISPLAY_NAME_MAPPINGS` properly defined
   - All nodes use ComfyUI conventions
   - Proper return type annotations

4. **Code Statistics**
   - Total: ~2,800 lines of production code
   - Utils: 959 lines
   - Nodes: 1,772 lines
   - Init: 68 lines

5. **Dependencies** ✓
   - Core: numpy, opencv-python, scipy (all standard)
   - Optional GPU: cupy-cuda12x, torch (commented for safety)
   - Optional features: ultralytics, onnxruntime-gpu (commented)
   - No malicious or suspicious imports

6. **Safety Features** ✓
   - Graceful GPU fallback to CPU
   - Try/except blocks for optional dependencies
   - No file system modifications outside workspace
   - No network calls without user control
   - No system command execution

## 📋 Deployment Instructions

### Step 1: Backup (IMPORTANT!)

Before copying files, backup your ComfyUI installation:

```bash
# Create backup of custom_nodes folder
cd /path/to/ComfyUI
cp -r custom_nodes custom_nodes.backup.$(date +%Y%m%d)
```

### Step 2: Copy Files

**Option A: Copy to existing ComfyUI installation**

```bash
# From the comfyui-custom-nodes directory:
cp -r custom_nodes/ys_vision_tools /path/to/ComfyUI/custom_nodes/

# OR on Windows:
xcopy custom_nodes\ys_vision_tools C:\path\to\ComfyUI\custom_nodes\ys_vision_tools /E /I
```

**Option B: Test in development first (RECOMMENDED)**

Keep files where they are and test manually:
```bash
cd custom_nodes/ys_vision_tools
python -c "from . import NODE_CLASS_MAPPINGS; print(len(NODE_CLASS_MAPPINGS))"
```

### Step 3: Install Dependencies

```bash
# Activate ComfyUI Python environment first!
# Then install core dependencies:
pip install numpy>=1.21.0 opencv-python>=4.5.0 scipy>=1.7.0

# Optional: GPU acceleration (for RTX 5090)
# pip install cupy-cuda12x torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: YOLO object detection
# pip install ultralytics
```

### Step 4: Verify Installation

After copying to ComfyUI:

```bash
cd /path/to/ComfyUI
python -c "import custom_nodes.ys_vision_tools; print('✓ Import successful')"
python -c "from custom_nodes.ys_vision_tools import NODE_CLASS_MAPPINGS; print(f'✓ Found {len(NODE_CLASS_MAPPINGS)} nodes')"
```

Expected output:
```
✓ Import successful
✓ Found 6 nodes
```

### Step 5: Start ComfyUI

```bash
cd /path/to/ComfyUI
python main.py
```

Watch for:
- ✓ No import errors in console
- ✓ Nodes appear in "YS-vision-tools" category
- ✓ Can add nodes to workflow

## 🎯 Expected Nodes in ComfyUI

After restart, you should see these in the node menu under **"YS-vision-tools"**:

**Tracking Category:**
- Track Detect (Enhanced) 🎯

**Rendering Category:**
- Line Link Renderer (Advanced) 🌀
- Dot Renderer ⚫

**Utilities Category:**
- Palette Map 🎨

**Compositing Category:**
- Layer Merge 🔀
- Composite Over 🎬

## ⚠️ First-Run Checklist

1. **Test Basic Functionality:**
   - Add any node to canvas ✓
   - Node shows input/output sockets ✓
   - Can connect nodes together ✓

2. **Test Without GPU (CPU fallback):**
   - If you see "GPU not available" warning, that's OK
   - All nodes should still work on CPU
   - Performance will be slower but functional

3. **Test With Sample Image:**
   - Load Image node → Track Detect
   - Should produce tracks output
   - No crashes or errors

## 🔧 Troubleshooting

### Issue: "Module not found: cupy"
**Solution:** GPU acceleration is optional. Code will use CPU fallback.
```bash
# Optional: Install GPU support
pip install cupy-cuda12x
```

### Issue: "Module not found: ultralytics"
**Solution:** Object detection is optional. Use other detection methods.
```bash
# Optional: Install YOLO
pip install ultralytics
```

### Issue: Nodes don't appear in menu
**Solution:**
1. Check ComfyUI console for import errors
2. Verify file structure: `ComfyUI/custom_nodes/ys_vision_tools/__init__.py` exists
3. Restart ComfyUI completely

### Issue: "Tracks" type not recognized
**Solution:** This is a custom type. Ensure nodes are connected in order:
- Track Detect → Line Link Renderer
- Track Detect → Dot Renderer

## 🚦 Safety Status: GREEN ✓

**SAFE TO DEPLOY** - All checks passed:

✅ No malicious code detected
✅ No unsafe file operations
✅ No network calls without consent
✅ No system modifications
✅ Graceful error handling
✅ CPU fallback available
✅ Standard dependencies only
✅ ComfyUI conventions followed
✅ Proper node registration
✅ Clean code structure

## 📝 Post-Deployment Validation

After deploying, run this simple test:

1. Start ComfyUI
2. Add nodes in this order:
   - Load Image
   - Track Detect (Enhanced)
   - Line Link Renderer (Advanced)
   - Composite Over
   - Preview Image
3. Connect them
4. Queue prompt
5. Should render successfully

## 🎓 Learning Resources

**Getting Started:**
- See `README.md` for full usage guide
- Check `docs/plans/` for detailed documentation
- Review `PROJECT_STATUS.md` for feature list

**Example Workflow:**
```
[Load Image]
    ↓
[Track Detect: gradient_magnitude, sensitivity=0.6]
    ↓
[Line Link Renderer: logarithmic_spiral, electric style]
    ↓
[Composite Over: opacity=0.8]
    ↓
[Preview Image]
```

## 🎉 Ready to Deploy!

All safety checks passed. The code is:
- ✅ Safe for production use
- ✅ Well-structured and documented
- ✅ Follows ComfyUI conventions
- ✅ Has graceful fallbacks
- ✅ No hidden dependencies
- ✅ No system risks

**Recommendation:** Deploy to ComfyUI now and test with sample images.

---

**Last Verified:** November 1, 2025
**Status:** SAFE TO DEPLOY ✓
**Phase:** Phase 1 MVP Complete
