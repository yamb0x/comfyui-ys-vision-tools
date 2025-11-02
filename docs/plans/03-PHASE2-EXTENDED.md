# Phase 2: Extended Features

## 🎯 Phase 2 Goal
Add advanced rendering capabilities and visual effects to the MVP system.

**Deliverables:**
1. BoundingBoxRenderer - draw boxes around tracked points
2. BlurRegionRenderer - apply blur effects to tracked regions
3. HUDTextRenderer - overlay technical/UI text
4. MVLookRenderer - machine vision aesthetic effects

## 📋 Implementation Tasks

### Task 9: BoundingBoxRenderer Node
**Files:**
- `custom_nodes/ys_vision/nodes/bbox_renderer.py`
- `tests/unit/test_bbox_renderer.py`
**Time:** 3 hours

#### 9.1 Write Tests First
```python
import numpy as np
import pytest
from custom_nodes.ys_vision.nodes.bbox_renderer import BoundingBoxRendererNode

def test_fixed_size_boxes():
    node = BoundingBoxRendererNode()
    tracks = np.array([[50, 50], [100, 100]])

    layer = node.execute(
        tracks=tracks,
        box_mode="fixed",
        width=20,
        height=20,
        stroke_px=2.0,
        fill_opacity=0.2,
        roundness=0.0,
        color=[1, 0, 0],
        image_width=200,
        image_height=200
    )[0]

    # Check box is drawn
    assert layer[40, 50, 0] > 0  # Top edge
    assert layer[60, 50, 0] > 0  # Bottom edge
    assert layer[50, 40, 0] > 0  # Left edge
    assert layer[50, 60, 0] > 0  # Right edge

def test_age_based_sizing():
    node = BoundingBoxRendererNode()
    tracks = np.array([[50, 50], [100, 100]])
    ages = np.array([5, 20])  # Different ages

    layer = node.execute(
        tracks=tracks,
        ages=ages,
        box_mode="from_age",
        stroke_px=2.0,
        color=[0, 1, 0],
        image_width=200,
        image_height=200
    )[0]

    # Older track should have larger box
    # Measure box sizes and compare
    pass
```

#### 9.2 Implement BoundingBoxRenderer
```python
import cv2
import numpy as np
from typing import Optional, Tuple, List

class BoundingBoxRendererNode:
    """Render bounding boxes around tracked points"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT",),
                "image_height": ("INT",),
                "box_mode": (["fixed", "from_radius", "from_age"],),
                "stroke_px": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0}),
                "fill_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "roundness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "tracks": ("TRACKS",),
                "boxes": ("BOXES",),
                "ages": ("AGES",),
                "palette": ("PALETTE",),
                "width": ("INT", {"default": 20}),
                "height": ("INT", {"default": 20}),
                "radius_px": ("FLOAT", {"default": 10}),
                "color": ("COLOR", {"default": [1, 1, 1]})
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Rendering"

    def execute(self, image_width, image_height, box_mode,
                stroke_px, fill_opacity, roundness, **kwargs):
        """Render bounding boxes to RGBA layer"""

        layer = np.zeros((image_height, image_width, 4), dtype=np.float32)

        # Get box positions and sizes
        boxes = self._compute_boxes(box_mode, **kwargs)

        for box in boxes:
            x, y, w, h = box[:4]
            color = box[4:7] if len(box) > 4 else kwargs.get('color', [1, 1, 1])

            if roundness > 0:
                self._draw_rounded_rect(layer, x, y, w, h,
                                       roundness, stroke_px,
                                       fill_opacity, color)
            else:
                self._draw_rect(layer, x, y, w, h,
                              stroke_px, fill_opacity, color)

        return (layer,)

    def _compute_boxes(self, mode: str, **kwargs) -> List[np.ndarray]:
        """Compute box dimensions based on mode"""
        if mode == "fixed":
            # Fixed size for all boxes
            tracks = kwargs.get('tracks', [])
            width = kwargs.get('width', 20)
            height = kwargs.get('height', 20)
            color = kwargs.get('color', [1, 1, 1])

            boxes = []
            for x, y in tracks:
                boxes.append([x - width/2, y - height/2,
                            width, height] + color)
            return boxes

        elif mode == "from_age":
            # Size based on track age
            tracks = kwargs.get('tracks', [])
            ages = kwargs.get('ages', np.ones(len(tracks)))
            base_size = kwargs.get('radius_px', 10)

            boxes = []
            for (x, y), age in zip(tracks, ages):
                # Logarithmic growth with age
                size = base_size * (1 + np.log1p(age) * 0.5)
                boxes.append([x - size, y - size,
                            size * 2, size * 2])
            return boxes

        elif mode == "from_radius":
            # Use specified radius
            tracks = kwargs.get('tracks', [])
            radius = kwargs.get('radius_px', 10)

            boxes = []
            for x, y in tracks:
                boxes.append([x - radius, y - radius,
                            radius * 2, radius * 2])
            return boxes

        return []

    def _draw_rect(self, layer: np.ndarray, x: float, y: float,
                   w: float, h: float, stroke: float,
                   fill_opacity: float, color: List[float]):
        """Draw rectangle with stroke and optional fill"""

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Fill first if requested
        if fill_opacity > 0:
            layer[y1:y2, x1:x2, :3] = color
            layer[y1:y2, x1:x2, 3] = fill_opacity

        # Draw stroke
        if stroke > 0:
            # Use cv2 for anti-aliased lines
            temp = np.zeros_like(layer)
            cv2.rectangle(temp, (x1, y1), (x2, y2),
                         (*color, 1.0), int(stroke))
            # Blend with layer
            alpha = temp[:, :, 3:4]
            layer[:, :, :3] = layer[:, :, :3] * (1 - alpha) + temp[:, :, :3] * alpha
            layer[:, :, 3] = np.maximum(layer[:, :, 3], temp[:, :, 3])

    def _draw_rounded_rect(self, layer: np.ndarray, x: float, y: float,
                          w: float, h: float, roundness: float,
                          stroke: float, fill_opacity: float,
                          color: List[float]):
        """Draw rounded rectangle"""
        # Calculate corner radius based on roundness (0-1)
        radius = min(w, h) * roundness * 0.5

        # Use cv2 to draw rounded rect
        # Or implement with circles at corners + rectangles
        pass
```

#### 9.3 Test & Commit
```bash
pytest tests/unit/test_bbox_renderer.py -v
git add -A
git commit -m "feat: Add BoundingBoxRenderer with multiple sizing modes"
```

---

### Task 10: BlurRegionRenderer Node
**Files:**
- `custom_nodes/ys_vision/nodes/blur_region_renderer.py`
- `tests/unit/test_blur_region_renderer.py`
**Time:** 3-4 hours

#### 10.1 Write Tests First
```python
def test_blur_at_track_positions():
    node = BlurRegionRendererNode()

    # Create test image with sharp features
    test_image = np.zeros((100, 100, 3))
    test_image[45:55, 45:55] = 1.0  # White square

    tracks = np.array([[50, 50]])  # Center of square

    layer = node.execute(
        image=test_image,
        tracks=tracks,
        radius_px=20,
        sigma_px=5.0,
        falloff=0.8,
        opacity=1.0
    )[0]

    # Check center is blurred
    assert layer[50, 50, 3] > 0  # Has alpha
    # Check edges have falloff
    assert layer[30, 50, 3] < layer[50, 50, 3]
```

#### 10.2 Implement BlurRegionRenderer
```python
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class BlurRegionRendererNode:
    """Apply blur effects to regions around tracked points"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius_px": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 100.0}),
                "sigma_px": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
                "falloff": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "tracks": ("TRACKS",),
                "boxes": ("BOXES",)
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Rendering"

    def execute(self, image, radius_px, sigma_px, falloff,
                opacity, **kwargs):
        """Create blur layer with masked regions"""

        h, w = image.shape[:2]
        layer = np.zeros((h, w, 4), dtype=np.float32)

        # Create blur mask from track positions
        mask = self._create_blur_mask(w, h, radius_px, falloff, **kwargs)

        # Apply gaussian blur to image
        blurred = gaussian_filter(image, sigma=sigma_px, axes=(0, 1))

        # Copy blurred regions to layer based on mask
        layer[:, :, :3] = blurred[:, :, :3]
        layer[:, :, 3] = mask * opacity

        return (layer,)

    def _create_blur_mask(self, width: int, height: int,
                         radius: float, falloff: float, **kwargs) -> np.ndarray:
        """Create mask for blur regions with falloff"""

        mask = np.zeros((height, width), dtype=np.float32)

        # Get positions to blur
        if 'tracks' in kwargs:
            positions = kwargs['tracks']
        elif 'boxes' in kwargs:
            # Use box centers
            boxes = kwargs['boxes']
            positions = boxes[:, :2] + boxes[:, 2:4] / 2
        else:
            return mask

        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:height, :width]

        for x, y in positions:
            # Distance from point
            dist = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)

            # Apply falloff
            if falloff > 0:
                # Smooth falloff using sigmoid
                local_mask = 1.0 / (1.0 + np.exp((dist - radius) / (radius * falloff * 0.1)))
            else:
                # Hard edge
                local_mask = (dist <= radius).astype(float)

            # Combine with max (overlapping regions)
            mask = np.maximum(mask, local_mask)

        return mask

    def _apply_masked_blur(self, image: np.ndarray, mask: np.ndarray,
                          sigma: float) -> np.ndarray:
        """Apply blur only to masked regions"""

        # This is more complex - need to blur then blend
        # based on mask to avoid bleeding
        blurred = gaussian_filter(image, sigma=sigma, axes=(0, 1))

        # Blend based on mask
        mask_3d = mask[:, :, np.newaxis]
        result = image * (1 - mask_3d) + blurred * mask_3d

        return result
```

#### 10.3 Test & Commit
```bash
pytest tests/unit/test_blur_region_renderer.py -v
git add -A
git commit -m "feat: Implement BlurRegionRenderer with falloff masks"
```

---

### Task 11: HUDTextRenderer Node
**Files:**
- `custom_nodes/ys_vision/nodes/hud_text_renderer.py`
- `tests/unit/test_hud_text_renderer.py`
- `custom_nodes/ys_vision/assets/fonts/generate_sdf.py`
**Time:** 4-5 hours

#### 11.1 Generate SDF Font Data
Create `custom_nodes/ys_vision/assets/fonts/generate_sdf.py`:
```python
"""Generate SDF (Signed Distance Field) font atlas"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_sdf_font():
    """Generate SDF font atlas for HUD text"""

    # Character set (ASCII printable)
    chars = ''.join(chr(i) for i in range(32, 127))

    # Create font bitmap
    font_size = 32
    atlas_size = 512

    # Use monospace font
    try:
        font = ImageFont.truetype("consolas.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Generate atlas
    img = Image.new('L', (atlas_size, atlas_size), 0)
    draw = ImageDraw.Draw(img)

    char_data = {}
    x, y = 0, 0
    row_height = 0

    for char in chars:
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if x + w > atlas_size:
            x = 0
            y += row_height
            row_height = 0

        draw.text((x, y), char, font=font, fill=255)

        char_data[char] = {
            'x': x, 'y': y,
            'width': w, 'height': h
        }

        x += w
        row_height = max(row_height, h)

    # Convert to SDF
    bitmap = np.array(img)
    sdf = compute_sdf(bitmap)

    # Save
    np.save('mono_sdf.npy', sdf)

    # Save character data
    import json
    with open('char_data.json', 'w') as f:
        json.dump(char_data, f)

def compute_sdf(bitmap: np.ndarray) -> np.ndarray:
    """Compute signed distance field from bitmap"""
    # Simple SDF computation
    # For production, use more sophisticated algorithm
    from scipy.ndimage import distance_transform_edt

    # Inside distances (white pixels)
    inside = distance_transform_edt(bitmap > 128)

    # Outside distances (black pixels)
    outside = distance_transform_edt(bitmap <= 128)

    # Combine into signed distance
    sdf = inside - outside

    # Normalize to 0-1
    sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min())

    return sdf

if __name__ == "__main__":
    generate_sdf_font()
```

#### 11.2 Write Tests First
```python
def test_text_rendering():
    node = HUDTextRendererNode()

    layer = node.execute(
        text="TRACKING: 042",
        font_px=16,
        position=(10, 10),
        opacity=1.0,
        color=[0, 1, 0],
        image_width=200,
        image_height=100
    )[0]

    # Check text is rendered
    assert layer[10:26, 10:150, 1].max() > 0  # Green channel
```

#### 11.3 Implement HUDTextRenderer
```python
import numpy as np
import json
from typing import Optional, Tuple, List

class HUDTextRendererNode:
    """Render HUD-style text overlays"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "font_px": ("INT", {"default": 16, "min": 8, "max": 64}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "image_width": ("INT",),
                "image_height": ("INT",),
            },
            "optional": {
                "position": ("POSITION", {"default": (10, 10)}),
                "tracks": ("TRACKS",),
                "ids": ("IDS",),
                "color": ("COLOR", {"default": [0, 1, 0]}),
                "grid": ("BOOLEAN", {"default": False}),
                "reticle": ("BOOLEAN", {"default": False}),
                "scramble_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
                "blink_hz": ("FLOAT", {"default": 0.0})
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Rendering"

    def __init__(self):
        # Load SDF font data
        self.font_atlas = None
        self.char_data = None
        self._load_font_data()

    def _load_font_data(self):
        """Load SDF font atlas and character mappings"""
        import os
        base_dir = os.path.dirname(__file__)
        font_dir = os.path.join(base_dir, '..', 'assets', 'fonts')

        try:
            self.font_atlas = np.load(os.path.join(font_dir, 'mono_sdf.npy'))
            with open(os.path.join(font_dir, 'char_data.json'), 'r') as f:
                self.char_data = json.load(f)
        except:
            # Fallback to basic rendering
            self.font_atlas = None
            self.char_data = None

    def execute(self, text, font_px, opacity, image_width, image_height, **kwargs):
        """Render HUD text elements"""

        layer = np.zeros((image_height, image_width, 4), dtype=np.float32)

        position = kwargs.get('position', (10, 10))
        color = kwargs.get('color', [0, 1, 0])
        scramble = kwargs.get('scramble_percent', 0.0)

        # Apply text scrambling effect
        if scramble > 0:
            text = self._scramble_text(text, scramble)

        # Render main text
        self._render_text(layer, text, position, font_px, color, opacity)

        # Optional: Add grid overlay
        if kwargs.get('grid', False):
            self._render_grid(layer, image_width, image_height, color, opacity * 0.3)

        # Optional: Add reticle
        if kwargs.get('reticle', False) and 'tracks' in kwargs:
            self._render_reticles(layer, kwargs['tracks'], color, opacity)

        # Optional: Add track IDs
        if 'tracks' in kwargs and 'ids' in kwargs:
            self._render_track_labels(layer, kwargs['tracks'],
                                     kwargs['ids'], font_px, color, opacity)

        return (layer,)

    def _render_text(self, layer: np.ndarray, text: str,
                    position: Tuple[int, int], size: int,
                    color: List[float], opacity: float):
        """Render text using SDF font or fallback"""

        if self.font_atlas is not None:
            # Use SDF rendering
            self._render_sdf_text(layer, text, position, size, color, opacity)
        else:
            # Fallback to OpenCV
            import cv2
            cv2.putText(layer, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       size / 16.0, (*color, opacity), 1)

    def _render_sdf_text(self, layer: np.ndarray, text: str,
                        position: Tuple[int, int], size: int,
                        color: List[float], opacity: float):
        """Render text using SDF font atlas"""

        x, y = position
        scale = size / 32.0  # Base font size is 32

        for char in text:
            if char not in self.char_data:
                x += size * 0.6  # Space for unknown char
                continue

            char_info = self.char_data[char]

            # Extract character from atlas
            cx, cy = char_info['x'], char_info['y']
            cw, ch = char_info['width'], char_info['height']

            char_sdf = self.font_atlas[cy:cy+ch, cx:cx+cw]

            # Scale to target size
            scaled_w = int(cw * scale)
            scaled_h = int(ch * scale)

            if scaled_w > 0 and scaled_h > 0:
                import cv2
                scaled = cv2.resize(char_sdf, (scaled_w, scaled_h))

                # Render to layer
                y1, y2 = y, min(y + scaled_h, layer.shape[0])
                x1, x2 = x, min(x + scaled_w, layer.shape[1])

                if y2 > y1 and x2 > x1:
                    # Apply SDF threshold for sharp edges
                    alpha = (scaled > 0.5).astype(float) * opacity
                    alpha = alpha[:y2-y1, :x2-x1]

                    # Blend with layer
                    for i in range(3):
                        layer[y1:y2, x1:x2, i] = color[i] * alpha
                    layer[y1:y2, x1:x2, 3] = np.maximum(layer[y1:y2, x1:x2, 3], alpha)

            x += scaled_w

    def _scramble_text(self, text: str, percent: float) -> str:
        """Apply glitch/scramble effect to text"""
        import random

        chars = list(text)
        num_scramble = int(len(chars) * percent / 100.0)

        for _ in range(num_scramble):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] != ' ':
                # Replace with random ASCII
                chars[idx] = chr(random.randint(33, 126))

        return ''.join(chars)

    def _render_grid(self, layer: np.ndarray, width: int, height: int,
                    color: List[float], opacity: float):
        """Render grid overlay"""

        grid_size = 50

        # Vertical lines
        for x in range(0, width, grid_size):
            layer[:, x:x+1, :3] = color
            layer[:, x:x+1, 3] = opacity

        # Horizontal lines
        for y in range(0, height, grid_size):
            layer[y:y+1, :, :3] = color
            layer[y:y+1, :, 3] = opacity

    def _render_reticles(self, layer: np.ndarray, tracks: np.ndarray,
                        color: List[float], opacity: float):
        """Render targeting reticles at track positions"""

        for x, y in tracks:
            x, y = int(x), int(y)

            # Draw crosshair
            size = 20
            # Horizontal line
            x1, x2 = max(0, x - size), min(layer.shape[1], x + size)
            layer[y:y+1, x1:x2, :3] = color
            layer[y:y+1, x1:x2, 3] = opacity

            # Vertical line
            y1, y2 = max(0, y - size), min(layer.shape[0], y + size)
            layer[y1:y2, x:x+1, :3] = color
            layer[y1:y2, x:x+1, 3] = opacity

            # Corner brackets
            bracket_size = 10
            thickness = 2

            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                # Horizontal part
                hx1 = x + dx * size
                hx2 = x + dx * (size - bracket_size)
                hy = y + dy * size

                if 0 <= hy < layer.shape[0]:
                    hx1, hx2 = sorted([hx1, hx2])
                    hx1 = max(0, hx1)
                    hx2 = min(layer.shape[1], hx2)
                    layer[hy:hy+thickness, hx1:hx2, :3] = color
                    layer[hy:hy+thickness, hx1:hx2, 3] = opacity
```

#### 11.4 Test & Commit
```bash
# Generate font data first
cd custom_nodes/ys_vision/assets/fonts
python generate_sdf.py

# Run tests
cd ../../../..
pytest tests/unit/test_hud_text_renderer.py -v
git add -A
git commit -m "feat: Add HUDTextRenderer with SDF font support"
```

---

### Task 12: MVLookRenderer Node (Machine Vision Look)
**Files:**
- `custom_nodes/ys_vision/nodes/mv_look_renderer.py`
- `tests/unit/test_mv_look_renderer.py`
**Time:** 3-4 hours

#### 12.1 Write Tests First
```python
def test_scanline_effect():
    node = MVLookRendererNode()

    # Test image
    image = np.ones((100, 100, 3)) * 0.5

    result = node.execute(
        image=image,
        scanline_intensity=0.5,
        scanline_spacing=2,
        chroma_offset_px=0,
        vignette=0.0,
        noise=0.0,
        as_layer=False
    )[0]

    # Check scanlines are present
    # Every other line should be darker
    assert result[0, 50, 0] != result[1, 50, 0]

def test_chromatic_aberration():
    node = MVLookRendererNode()

    # Image with white center
    image = np.zeros((100, 100, 3))
    image[40:60, 40:60] = 1.0

    result = node.execute(
        image=image,
        scanline_intensity=0.0,
        chroma_offset_px=2,
        vignette=0.0,
        noise=0.0,
        as_layer=False
    )[0]

    # Check RGB channels are offset
    # Red should be shifted left, blue right
    assert result[50, 48, 0] > result[50, 52, 0]  # Red
    assert result[50, 52, 2] > result[50, 48, 2]  # Blue
```

#### 12.2 Implement MVLookRenderer
```python
import numpy as np
from scipy.ndimage import shift

class MVLookRendererNode:
    """Apply machine vision aesthetic effects"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scanline_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "chroma_offset_px": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "vignette": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "noise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5}),
                "as_layer": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "opacity": ("FLOAT", {"default": 1.0}),
                "scanline_spacing": ("INT", {"default": 2, "min": 1, "max": 10}),
                "color_tint": ("COLOR", {"default": [0.0, 1.0, 0.8]})  # Cyan tint
            }
        }

    RETURN_TYPES = ("IMAGE", "LAYER")
    RETURN_NAMES = ("image", "layer")
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Effects"

    def execute(self, image, scanline_intensity, chroma_offset_px,
                vignette, noise, as_layer, **kwargs):
        """Apply machine vision look effects"""

        result = image.copy()

        # Apply effects in sequence
        if scanline_intensity > 0:
            result = self._apply_scanlines(result, scanline_intensity,
                                          kwargs.get('scanline_spacing', 2))

        if chroma_offset_px > 0:
            result = self._apply_chromatic_aberration(result, chroma_offset_px)

        if vignette > 0:
            result = self._apply_vignette(result, vignette)

        if noise > 0:
            result = self._apply_noise(result, noise)

        # Apply color tint
        tint = kwargs.get('color_tint', [0.0, 1.0, 0.8])
        if tint != [1.0, 1.0, 1.0]:
            result = self._apply_color_tint(result, tint)

        if as_layer:
            # Return as RGBA layer
            opacity = kwargs.get('opacity', 1.0)
            layer = np.zeros((result.shape[0], result.shape[1], 4))
            layer[:, :, :3] = result
            layer[:, :, 3] = opacity
            return (image, layer)
        else:
            # Return modified image
            return (result, None)

    def _apply_scanlines(self, image: np.ndarray, intensity: float,
                        spacing: int) -> np.ndarray:
        """Add CRT-style scanlines"""

        result = image.copy()
        h, w = image.shape[:2]

        # Create scanline pattern
        for y in range(0, h, spacing):
            # Darken every nth line
            result[y, :] *= (1.0 - intensity)

            # Optional: Add subtle bright line after dark
            if y + 1 < h and spacing > 1:
                result[y + 1, :] *= (1.0 + intensity * 0.2)
                result[y + 1, :] = np.clip(result[y + 1, :], 0, 1)

        return result

    def _apply_chromatic_aberration(self, image: np.ndarray,
                                   offset: float) -> np.ndarray:
        """Simulate lens chromatic aberration"""

        result = image.copy()

        # Shift red channel left
        result[:, :, 0] = shift(image[:, :, 0], [0, -offset], mode='nearest')

        # Keep green centered
        # result[:, :, 1] = image[:, :, 1]

        # Shift blue channel right
        result[:, :, 2] = shift(image[:, :, 2], [0, offset], mode='nearest')

        return result

    def _apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Add vignette darkening at edges"""

        h, w = image.shape[:2]
        result = image.copy()

        # Create radial gradient
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]

        # Distance from center (normalized)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_dist

        # Smooth falloff
        vignette_mask = 1.0 - (dist**2 * strength)
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # Apply to all channels
        for i in range(3):
            result[:, :, i] *= vignette_mask

        return result

    def _apply_noise(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Add film grain / sensor noise"""

        # Generate noise
        noise = np.random.normal(0, amount, image.shape)

        # Add noise and clip
        result = image + noise
        result = np.clip(result, 0, 1)

        return result

    def _apply_color_tint(self, image: np.ndarray, tint: List[float]) -> np.ndarray:
        """Apply color tint/filter"""

        result = image.copy()

        for i in range(3):
            result[:, :, i] *= tint[i]

        return np.clip(result, 0, 1)

    def _add_digital_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add digital glitch artifacts"""

        result = image.copy()
        h, w = image.shape[:2]

        # Random horizontal glitch lines
        num_glitches = np.random.randint(0, 3)

        for _ in range(num_glitches):
            y = np.random.randint(0, h)
            height = np.random.randint(1, 5)
            shift_amount = np.random.randint(-10, 10)

            y1 = max(0, y)
            y2 = min(h, y + height)

            # Shift RGB channels differently
            for i in range(3):
                shift_val = shift_amount + np.random.randint(-2, 2)
                result[y1:y2, :, i] = np.roll(result[y1:y2, :, i],
                                             shift_val, axis=1)

        return result
```

#### 12.3 Test & Commit
```bash
pytest tests/unit/test_mv_look_renderer.py -v
git add -A
git commit -m "feat: Implement MVLookRenderer with CRT and glitch effects"
```

---

## 🧪 Phase 2 Testing Protocol

### Integration Test for Phase 2
Create `tests/integration/test_phase2_pipeline.py`:
```python
import numpy as np
import pytest

def test_complete_phase2_workflow():
    """Test all Phase 2 nodes work together"""

    # Create test image
    test_image = np.random.rand(480, 640, 3)

    # Simulate tracked points
    tracks = np.array([
        [100, 100], [200, 150],
        [300, 200], [400, 250]
    ])

    # Test each renderer
    # 1. BoundingBoxRenderer
    bbox_node = BoundingBoxRendererNode()
    bbox_layer = bbox_node.execute(
        tracks=tracks,
        box_mode="fixed",
        width=30, height=30,
        stroke_px=2.0,
        image_width=640,
        image_height=480
    )[0]

    assert bbox_layer.shape == (480, 640, 4)

    # 2. BlurRegionRenderer
    blur_node = BlurRegionRendererNode()
    blur_layer = blur_node.execute(
        image=test_image,
        tracks=tracks,
        radius_px=25,
        sigma_px=5.0,
        falloff=0.7,
        opacity=0.8
    )[0]

    assert blur_layer.shape == (480, 640, 4)

    # 3. HUDTextRenderer
    hud_node = HUDTextRendererNode()
    hud_layer = hud_node.execute(
        text="TRACKING ACTIVE",
        font_px=14,
        tracks=tracks,
        ids=np.array([1, 2, 3, 4]),
        reticle=True,
        image_width=640,
        image_height=480
    )[0]

    assert hud_layer.shape == (480, 640, 4)

    # 4. MVLookRenderer
    mv_node = MVLookRendererNode()
    processed_image, mv_layer = mv_node.execute(
        image=test_image,
        scanline_intensity=0.2,
        chroma_offset_px=1.5,
        vignette=0.3,
        noise=0.05,
        as_layer=True,
        opacity=0.6
    )

    assert processed_image.shape == test_image.shape
    assert mv_layer.shape == (480, 640, 4)

    # Test layer merging
    merge_node = LayerMergeNode()
    final_layer = merge_node.execute(
        layers=[bbox_layer, blur_layer, hud_layer, mv_layer],
        blend_modes=["normal", "normal", "add", "screen"]
    )[0]

    assert final_layer.shape == (480, 640, 4)

    # Final composite
    comp_node = CompositeOverNode()
    final_image = comp_node.execute(
        image=test_image,
        layer=final_layer
    )[0]

    assert final_image.shape == test_image.shape
    print("✅ Phase 2 integration test passed!")
```

### Performance Benchmark
Create `tests/performance/test_phase2_performance.py`:
```python
import time
import numpy as np

def benchmark_node(node_class, execute_params, iterations=100):
    """Benchmark a node's execution time"""

    node = node_class()
    times = []

    for _ in range(iterations):
        start = time.time()
        result = node.execute(**execute_params)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"{node_class.__name__}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    return avg_time

def test_phase2_performance():
    """Benchmark all Phase 2 nodes"""

    # Standard test data
    image = np.random.rand(1080, 1920, 3)
    tracks = np.random.rand(200, 2) * [1920, 1080]

    # Benchmark each node
    benchmarks = {}

    # BoundingBoxRenderer
    benchmarks['bbox'] = benchmark_node(
        BoundingBoxRendererNode,
        {'tracks': tracks, 'box_mode': 'fixed',
         'width': 20, 'height': 20, 'stroke_px': 2,
         'image_width': 1920, 'image_height': 1080}
    )

    # BlurRegionRenderer
    benchmarks['blur'] = benchmark_node(
        BlurRegionRendererNode,
        {'image': image, 'tracks': tracks,
         'radius_px': 20, 'sigma_px': 5, 'falloff': 0.8}
    )

    # HUDTextRenderer
    benchmarks['hud'] = benchmark_node(
        HUDTextRendererNode,
        {'text': 'TEST', 'font_px': 16,
         'image_width': 1920, 'image_height': 1080}
    )

    # MVLookRenderer
    benchmarks['mvlook'] = benchmark_node(
        MVLookRendererNode,
        {'image': image, 'scanline_intensity': 0.3,
         'chroma_offset_px': 2, 'vignette': 0.3, 'as_layer': False}
    )

    # Check performance targets
    assert benchmarks['bbox'] < 0.010  # <10ms
    assert benchmarks['blur'] < 0.050  # <50ms
    assert benchmarks['hud'] < 0.005   # <5ms
    assert benchmarks['mvlook'] < 0.030  # <30ms

    print("✅ All Phase 2 nodes meet performance targets!")
```

## 📊 Phase 2 Success Metrics

### Completion Checklist:
- [ ] BoundingBoxRenderer implemented with 3 sizing modes
- [ ] BlurRegionRenderer with smooth falloff masks
- [ ] HUDTextRenderer with SDF font and effects
- [ ] MVLookRenderer with all visual effects
- [ ] All nodes integrate with Phase 1 pipeline
- [ ] Performance targets met (1080p @ 15+ fps)
- [ ] Test coverage > 85%
- [ ] Visual quality matches reference images
- [ ] Memory usage stable over 1000 frames

## 🎬 Next Steps
- Continue to `04-PHASE3-OPTIMIZATION.md` for GPU acceleration
- Or continue to `05-PHASE4-ADVANCED.md` for clustering and advanced features