# Color Picker Implementation Guide

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools for ComfyUI
**Status:** Active Implementation Pattern

---

## Overview

This document describes the standardized color picker implementation pattern used throughout YS-vision-tools. All nodes with color parameters use ComfyUI's native `COLOR` input type for visual color selection.

**Key Features:**
- Visual color picker UI (click color swatch)
- Supports HEX colors: `"#ffffff"`, `"#ff0000"`, `"#00ff00"`
- Supports named colors: `"red"`, `"orange"`, `"cyan"`, `"white"`
- Backward compatible with legacy float lists: `[1.0, 0.5, 0.0]`
- Separate alpha slider for transparency control
- Centralized parsing with `normalize_color_to_rgba01()`

---

## Reference Implementation

**Primary Reference:** `nodes/bbox_renderer.py` (complete working example)
**Utility Function:** `utils/color_utils.py` - Color parsing logic
**Export Location:** `utils/__init__.py` - Exports `normalize_color_to_rgba01`

---

## Implementation Pattern

### Step 1: Import Color Utility

```python
from ..utils import (
    create_rgba_layer,
    numpy_to_comfyui,
    normalize_color_to_rgba01  # Color parsing function
)
```

### Step 2: Update INPUT_TYPES

**Move color to `required`, add alpha to `optional`:**

```python
@classmethod
def INPUT_TYPES(cls) -> Dict[str, Any]:
    return {
        "required": {
            # ... other required parameters ...

            "color": ("COLOR", {
                "default": "#ffffff",
                "tooltip": "Click the color swatch to open the visual color picker"
            }),
        },
        "optional": {
            # ... other optional parameters ...

            "alpha": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Transparency level (0=invisible, 1=opaque)"
            }),
        }
    }
```

**Key Changes:**
- `("STRING", {"default": "1.0,1.0,1.0"})` → `("COLOR", {"default": "#ffffff"})`
- Convert defaults to HEX format (white: `"#ffffff"`, red: `"#ff0000"`, blue: `"#0000ff"`)
- Add separate `alpha` slider for transparency control

### Step 3: Update execute() Signature

```python
def execute(self, image_width, image_height, ..., color, **kwargs):
    """
    Execute node with color input

    Args:
        color: COLOR input (hex string like "#ffffff" or named color like "red")
               Automatically parsed by normalize_color_to_rgba01()
        **kwargs: Optional parameters including 'alpha' for transparency
    """
    # Parse color once at top level
    alpha = kwargs.get('alpha', 1.0)
    rgba = normalize_color_to_rgba01(color, alpha)

    print(f"[YS-NODE] Parsed color: {color} -> RGBA: {rgba}")

    # Pass color to rendering methods
    layer = self._render_single_frame(..., color, **kwargs)
    return (numpy_to_comfyui(layer),)
```

**Key Points:**
- Add `color` parameter to function signature
- Parse color early with `normalize_color_to_rgba01()`
- Log parsed color for debugging
- Pass `color` parameter to internal methods

### Step 4: Parse Color in Rendering Method

```python
def _render_single_frame(self, ..., color, **kwargs):
    """Internal rendering method

    Args:
        color: COLOR input (hex/named color) passed from execute()
        **kwargs: Optional parameters including 'alpha' override
    """

    # Parse color using centralized utility
    alpha = kwargs.get('alpha', 1.0)  # Get alpha from optional parameter
    rgba = normalize_color_to_rgba01(color, alpha)
    color_rgb = rgba[:3]  # Extract RGB tuple for rendering functions

    # Use color_rgb in rendering operations
    # Example: cv2.rectangle(..., color_rgb, ...)
    # Example: layer[..., :3] = color_rgb
```

**Key Points:**
- Parse color in the rendering method (closest to usage)
- Extract RGB tuple with `rgba[:3]` for OpenCV/NumPy operations
- Use full `rgba` for operations that need alpha channel

### Step 5: Handle Batch Processing

```python
def execute(self, ..., color, **kwargs):
    """Execute with batch support"""

    tracks = kwargs.get('tracks', np.array([]))

    # Check if batch mode (list of track arrays)
    if isinstance(tracks, list):
        print(f"[YS-NODE] BATCH MODE: {len(tracks)} frames")
        batch_layers = []

        for i, frame_tracks in enumerate(tracks):
            # Parse color ONCE outside loop (not per-frame)
            # Color is the same for all frames
            frame_kwargs = kwargs.copy()
            frame_kwargs['tracks'] = frame_tracks

            layer = self._render_single_frame(..., color, **frame_kwargs)
            batch_layers.append(layer)

        # Stack into batch
        batch_result = np.stack(batch_layers, axis=0)
        import torch
        return (torch.from_numpy(batch_result.astype(np.float32)),)

    # Single frame mode
    layer = self._render_single_frame(..., color, **kwargs)
    return (numpy_to_comfyui(layer),)
```

**Key Points:**
- Color parsing should happen in `_render_single_frame()`, not in loop
- Same color applies to all frames in batch
- Don't re-parse color per-frame (performance optimization)

---

## Color Utility Function

### normalize_color_to_rgba01()

**Location:** `utils/color_utils.py`

**Signature:**
```python
def normalize_color_to_rgba01(color_input, alpha: float = 1.0) -> tuple:
    """
    Convert color input to normalized RGBA tuple (0.0-1.0 range)

    Args:
        color_input: Can be:
            - HEX string: "#ffffff", "#ff0000"
            - Named color: "red", "blue", "white", "cyan"
            - Legacy string: "1.0,0.5,0.0" (backward compatibility)
            - Float list: [1.0, 0.5, 0.0] (backward compatibility)
        alpha: Alpha value (0.0-1.0), defaults to 1.0 (opaque)

    Returns:
        tuple: (R, G, B, A) with values in 0.0-1.0 range

    Examples:
        >>> normalize_color_to_rgba01("#ff0000", 0.8)
        (1.0, 0.0, 0.0, 0.8)

        >>> normalize_color_to_rgba01("blue", 1.0)
        (0.0, 0.0, 1.0, 1.0)

        >>> normalize_color_to_rgba01("1.0,0.5,0.0", 0.5)
        (1.0, 0.5, 0.0, 0.5)
    """
```

**Supported Formats:**

1. **HEX Colors (Primary Format)**
   ```python
   "#ffffff"  # White
   "#ff0000"  # Red
   "#00ff00"  # Green
   "#0000ff"  # Blue
   "#ff8800"  # Orange
   ```

2. **Named Colors**
   ```python
   "red"      # (1.0, 0.0, 0.0)
   "green"    # (0.0, 1.0, 0.0)
   "blue"     # (0.0, 0.0, 1.0)
   "white"    # (1.0, 1.0, 1.0)
   "black"    # (0.0, 0.0, 0.0)
   "cyan"     # (0.0, 1.0, 1.0)
   "magenta"  # (1.0, 0.0, 1.0)
   "yellow"   # (1.0, 1.0, 0.0)
   "orange"   # (1.0, 0.5, 0.0)
   ```

3. **Legacy Formats (Backward Compatibility)**
   ```python
   "1.0,0.5,0.0"        # Comma-separated string
   [1.0, 0.5, 0.0]      # Float list
   (1.0, 0.5, 0.0)      # Float tuple
   ```

**Auto-Detection:**
- Values 0.0-1.0: Treated as normalized floats
- Values > 1.0: Auto-detects as 0-255 range, converts to 0.0-1.0

---

## Common Patterns

### Pattern 1: Single Color Parameter (Most Common)

**Applies to:** Most rendering nodes (BBox Renderer, Dot Renderer, Curved Line, etc.)

```python
# INPUT_TYPES
"required": {
    "color": ("COLOR", {"default": "#ffffff"}),
},
"optional": {
    "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
}

# execute() method
def execute(self, ..., color, **kwargs):
    alpha = kwargs.get('alpha', 1.0)
    rgba = normalize_color_to_rgba01(color, alpha)
    color_rgb = rgba[:3]
    # Use color_rgb in rendering
```

### Pattern 2: Multiple Color Parameters

**Applies to:** Nodes with text + stroke (Text On Tracks), gradient effects

```python
# INPUT_TYPES
"required": {
    "color": ("COLOR", {"default": "#ffffff"}),
    "stroke_color": ("COLOR", {"default": "#000000"}),
},
"optional": {
    "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
}

# execute() method
def execute(self, ..., color, stroke_color, opacity, **kwargs):
    # Parse both colors
    text_rgba = normalize_color_to_rgba01(color, opacity)
    stroke_rgba = normalize_color_to_rgba01(stroke_color, 1.0)  # Stroke always opaque

    # Use in rendering
    text_color = text_rgba[:3]
    stroke_color_rgb = stroke_rgba[:3]
```

### Pattern 3: Gradient Colors (Optional)

**Applies to:** Line renderers with gradient styles

```python
# INPUT_TYPES
"optional": {
    "gradient_start_color": ("COLOR", {"default": "#ffffff"}),
    "gradient_end_color": ("COLOR", {"default": "#0080ff"}),
    "gradient_alpha": ("FLOAT", {"default": 1.0}),
}

# execute() method - only parse if gradient style used
def _apply_line_style(self, ..., line_style, **kwargs):
    if line_style == "gradient_fade":
        gradient_alpha = kwargs.get('gradient_alpha', 1.0)
        start_rgba = normalize_color_to_rgba01(
            kwargs.get('gradient_start_color', '#ffffff'),
            gradient_alpha
        )
        end_rgba = normalize_color_to_rgba01(
            kwargs.get('gradient_end_color', '#0080ff'),
            gradient_alpha
        )
        # Apply gradient...
```

---

## Nodes with Color Picker Implementation

### Fully Implemented (Using COLOR Type)

1. **BBox Renderer** - Single color + alpha slider
   - File: `nodes/bbox_renderer.py`
   - Parameters: `color` (COLOR), `alpha` (FLOAT)
   - Status: ✅ Complete

2. **Text On Tracks** - Text color + stroke color
   - File: `nodes/text_on_tracks.py`
   - Parameters: `color` (COLOR), `stroke_color` (COLOR), `opacity` (FLOAT)
   - Status: ✅ Complete

3. **Dot Renderer** - Single color + alpha slider
   - File: `nodes/dot_renderer.py`
   - Parameters: `color` (COLOR), `alpha` (FLOAT)
   - Status: ✅ Complete

### Remaining Nodes (Need Implementation)

4. **Curved Line Renderer**
   - File: `nodes/curved_line_renderer.py`
   - Current: `"color": ("STRING", {"default": "1.0,1.0,1.0"})`
   - Pattern: Single color + alpha slider

5. **Graph Line Renderer**
   - File: `nodes/graph_line_renderer.py`
   - Current: `"color": ("STRING", {"default": "1.0,1.0,1.0"})`
   - Pattern: Single color + alpha slider

6. **Physics Line Renderer**
   - File: `nodes/physics_line_renderer.py`
   - Current: `"color": ("STRING", {"default": "0.3,0.7,1.0"})`
   - Pattern: Single color + alpha slider

7. **Line Link Renderer (Advanced)** - Gradient support
   - File: `nodes/line_link_renderer.py`
   - Current: `gradient_start_color`, `gradient_end_color` (STRING)
   - Pattern: Two optional color pickers + gradient_alpha

---

## Testing Checklist

### Code Verification (Before User Testing)

- [ ] `normalize_color_to_rgba01` imported from utils
- [ ] `INPUT_TYPES` has `("COLOR", {...})` for color parameter(s)
- [ ] Alpha slider added (or opacity parameter for multi-color nodes)
- [ ] `execute()` method signature includes `color` parameter
- [ ] Color parsing uses `normalize_color_to_rgba01(color, alpha)`
- [ ] All color usage updated to use parsed `rgba` or `rgba[:3]`
- [ ] Default colors converted to HEX (e.g., `"1.0,1.0,1.0"` → `"#ffffff"`)
- [ ] Debug logging added: `print(f"[YS-NODE] Parsed color: {color} -> RGBA: {rgba}")`

**Verification Commands:**
```bash
# Verify function is imported
grep -n "normalize_color_to_rgba01" "D:\path\to\node.py"

# Verify COLOR type is used
grep -n '"COLOR"' "D:\path\to\node.py"

# Check line count changed
wc -l "D:\path\to\node.py"
```

### ComfyUI Testing (After Copying Files)

- [ ] Copy updated file from D: to F: drive
- [ ] Restart ComfyUI server
- [ ] Add node to workflow
- [ ] **VERIFY:** Color parameter shows color swatch button (NOT text input)
- [ ] **VERIFY:** Clicking swatch opens visual color picker
- [ ] **VERIFY:** Selecting color updates swatch in real-time
- [ ] **VERIFY:** Alpha slider works (0.0 = invisible, 1.0 = opaque)
- [ ] **VERIFY:** Console shows parsed color log
- [ ] **VERIFY:** Visual output shows correct color
- [ ] **VERIFY:** Test with video batch (50+ frames)
- [ ] **VERIFY:** Backward compatibility with legacy string format

---

## Common Pitfalls

### Pitfall 1: Wrong Parameter Section
**Problem:** Color in "optional" but should be in "required"
**Solution:** Color is almost always required, alpha is optional

### Pitfall 2: Forgot to Update execute() Signature
**Problem:** ComfyUI passes color but function doesn't accept it
**Solution:** Add `color` parameter to execute() signature

### Pitfall 3: Parsing Color Per-Frame in Batch
**Problem:** Performance degrades, color parsed 50+ times for video
**Solution:** Parse once in `_render_single_frame()`, not in batch loop

### Pitfall 4: Wrong RGB Extraction
**Problem:** Using `rgba` directly where RGB tuple expected
**Solution:** Use `rgba[:3]` to extract RGB for OpenCV/NumPy operations

### Pitfall 5: Missing Debug Logging
**Problem:** Can't verify color picker values received correctly
**Solution:** Always add: `print(f"[YS-NODE] Parsed color: {color} -> RGBA: {rgba}")`

### Pitfall 6: Default Color Still String Format
**Problem:** Default shows `"1.0,1.0,1.0"` instead of HEX
**Solution:** Convert all defaults: white=`"#ffffff"`, red=`"#ff0000"`, etc.

---

## File Header Documentation

Add this to node file headers when color picker is implemented:

```python
"""
Node Name - Description

🎨 COLOR PICKER SUPPORT:
- Visual color picker UI (click the color swatch in ComfyUI)
- Supports HEX colors: "#ffffff", "#ff0000", "#00ff00"
- Supports named colors: "red", "orange", "cyan", "white"
- Backward compatible with legacy float lists: [1.0, 0.5, 0.0]
- Separate alpha slider for transparency control

Author: Yambo Studio
Part of: YS-vision-tools
"""
```

---

## References

- **BBox Renderer Reference:** `nodes/bbox_renderer.py` - Complete implementation example
- **Text On Tracks Reference:** `nodes/text_on_tracks.py` - Multi-color example
- **Color Utility:** `utils/color_utils.py` - Parsing logic
- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.md` - Node structure patterns
- **Project Rules:** `CLAUDE.md` - Development guidelines

---

**Last Updated:** 2025-11-08
**Project:** YS-vision-tools
**Developer:** Yambo Studio
