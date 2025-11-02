# Phase 1: MVP Implementation (Core Tracking & Basic Rendering)

## 🎯 Phase 1 Goal
Build the minimum viable product: track points in video and render colored dots and lines.

**Deliverables:**
1. TrackDetect node - finds points in frames
2. PaletteMap node - assigns colors to tracks
3. DotRenderer node - draws colored dots
4. LineLinkRenderer node - connects dots with lines
5. LayerMerge node - combines layers
6. CompositeOver node - applies to original image

## 📋 Implementation Tasks

### Task 1: Common Utilities Module
**File:** `custom_nodes/ys_vision/utils/common.py`
**Time:** 2-3 hours

#### 1.1 Write Tests First (TDD)
Create `tests/unit/test_common_utils.py`:
```python
import numpy as np
import pytest
from custom_nodes.ys_vision.utils.common import (
    ensure_rgb, ensure_rgba, tensor_to_numpy,
    numpy_to_tensor, normalize_points, clip_points
)

def test_ensure_rgb():
    # Grayscale to RGB
    gray = np.random.rand(100, 100)
    rgb = ensure_rgb(gray)
    assert rgb.shape == (100, 100, 3)

    # Already RGB
    rgb_in = np.random.rand(100, 100, 3)
    rgb_out = ensure_rgb(rgb_in)
    assert rgb_out.shape == (100, 100, 3)

def test_normalize_points():
    points = np.array([[100, 200], [300, 400]])
    norm = normalize_points(points, 800, 600)
    assert np.all(norm >= 0) and np.all(norm <= 1)
```

#### 1.2 Implement Common Utilities
```python
"""Common utilities for YS-Vision nodes"""
import numpy as np
import torch
from typing import Tuple, Union

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any image format to RGB"""
    # Implementation here
    pass

def ensure_rgba(image: np.ndarray) -> np.ndarray:
    """Convert any image format to RGBA"""
    # Implementation here
    pass

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI tensor to numpy array"""
    # Implementation here
    pass
```

#### 1.3 Commit
```bash
git add -A
git commit -m "feat: Add common utilities module with image format conversions"
```

---

### Task 2: TrackDetect Node (Core)
**Files:**
- `custom_nodes/ys_vision/nodes/track_detect.py`
- `tests/unit/test_track_detect.py`
**Time:** 4-5 hours

#### 2.1 Write Tests First
Create `tests/unit/test_track_detect.py`:
```python
import numpy as np
import pytest
from custom_nodes.ys_vision.nodes.track_detect import TrackDetectNode

class TestTrackDetectNode:
    def test_input_types(self):
        inputs = TrackDetectNode.INPUT_TYPES()
        assert "required" in inputs
        assert "IMAGE" in inputs["required"]

    def test_corner_detection(self):
        node = TrackDetectNode()
        # Create test image with corners
        image = np.zeros((100, 100, 3))
        image[20:30, 20:30] = 1.0  # White square

        tracks, ids, conf, _, state = node.execute(
            image=image,
            threshold_mode="global",
            threshold=0.5,
            detector="corner",
            points_per_frame=10
        )

        assert len(tracks) > 0
        assert len(ids) == len(tracks)
        assert len(conf) == len(tracks)
```

#### 2.2 Implement TrackDetect Node
Create `custom_nodes/ys_vision/nodes/track_detect.py`:
```python
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class TrackDetectNode:
    """Detect and track points across video frames"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_mode": (["global", "adaptive"],),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "blur": ("INT", {"default": 3, "min": 0, "max": 15}),
                "normalize": ("BOOLEAN", {"default": True}),
                "detector": (["corner", "blob"],),
                "points_per_frame": ("INT", {"default": 200, "min": 1, "max": 1000}),
            },
            "optional": {
                "state": ("TRACK_STATE",),
                "klt_enable": ("BOOLEAN", {"default": False}),
                "max_track_age": ("INT", {"default": 30}),
                "bbox_radius": ("FLOAT", {"default": 0})
            }
        }

    RETURN_TYPES = ("TRACKS", "IDS", "CONFIDENCE", "BOXES", "TRACK_STATE")
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Tracking"

    def execute(self, image, threshold_mode, threshold, blur,
                normalize, detector, points_per_frame, **kwargs):
        """Main execution function"""
        # Convert image format
        # Apply preprocessing (blur, normalize)
        # Apply thresholding
        # Detect points (corner or blob)
        # Optional: KLT tracking
        # Return track data
        pass
```

#### 2.3 Implement Detection Methods
Add to `track_detect.py`:
```python
def _preprocess_image(self, image: np.ndarray, blur: int,
                     normalize: bool) -> np.ndarray:
    """Preprocess image for detection"""
    # Convert to grayscale
    # Apply gaussian blur
    # Normalize if requested
    pass

def _detect_corners(self, image: np.ndarray,
                   max_points: int) -> np.ndarray:
    """Detect corners using Shi-Tomasi"""
    # Use cv2.goodFeaturesToTrack
    pass

def _detect_blobs(self, image: np.ndarray,
                 max_points: int) -> np.ndarray:
    """Detect blobs using SimpleBlobDetector"""
    # Configure and use cv2.SimpleBlobDetector
    pass
```

#### 2.4 Test & Commit
```bash
pytest tests/unit/test_track_detect.py -v
git add -A
git commit -m "feat: Implement TrackDetect node with corner and blob detection"
```

---

### Task 3: PaletteMap Node
**Files:**
- `custom_nodes/ys_vision/nodes/palette_map.py`
- `tests/unit/test_palette_map.py`
**Time:** 3 hours

#### 3.1 Write Tests First
```python
class TestPaletteMapNode:
    def test_color_by_id(self):
        node = PaletteMapNode()
        ids = np.array([1, 2, 3, 1, 2])
        palette, colors = node.execute(
            ids=ids,
            mode="by_id",
            palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        assert colors.shape == (5, 3)
        # ID 1 should always get same color
        assert np.array_equal(colors[0], colors[3])
```

#### 3.2 Implement PaletteMap
```python
class PaletteMapNode:
    """Map tracks to colors based on various strategies"""

    def execute(self, ids, mode, palette=None, **kwargs):
        """Assign colors based on mode"""
        if mode == "by_id":
            # Hash ID to palette index
            pass
        elif mode == "by_age":
            # Map age to color gradient
            pass
        # etc.
```

#### 3.3 Commit
```bash
pytest tests/unit/test_palette_map.py -v
git add -A
git commit -m "feat: Add PaletteMap node for color assignment strategies"
```

---

### Task 4: DotRenderer Node
**Files:**
- `custom_nodes/ys_vision/nodes/dot_renderer.py`
- `tests/unit/test_dot_renderer.py`
**Time:** 3 hours

#### 4.1 Write Tests First
```python
def test_dot_rendering():
    node = DotRendererNode()
    tracks = np.array([[50, 50], [100, 100]])
    colors = np.array([[1, 0, 0], [0, 1, 0]])

    layer = node.execute(
        tracks=tracks,
        palette=colors,
        size_px=5.0,
        softness=0.5,
        opacity=1.0,
        image_width=200,
        image_height=200
    )[0]

    # Check dots are rendered at positions
    assert layer[50, 50, 0] > 0  # Red channel at first dot
    assert layer[100, 100, 1] > 0  # Green channel at second dot
```

#### 4.2 Implement DotRenderer
```python
class DotRendererNode:
    """Render dots at track positions"""

    def execute(self, tracks, size_px, softness, opacity,
                image_width, image_height, **kwargs):
        """Render dots to RGBA layer"""
        layer = np.zeros((image_height, image_width, 4))

        for i, (x, y) in enumerate(tracks):
            # Draw dot with anti-aliasing
            # Apply softness (gradient falloff)
            # Apply color from palette
            pass

        return (layer,)
```

#### 4.3 Add Anti-aliasing
```python
def _draw_soft_circle(self, layer: np.ndarray, x: float, y: float,
                      radius: float, color: np.ndarray, softness: float):
    """Draw anti-aliased soft circle"""
    # Create mesh grid around point
    # Calculate distance from center
    # Apply gaussian falloff
    # Blend with existing layer
    pass
```

#### 4.4 Commit
```bash
pytest tests/unit/test_dot_renderer.py -v
git add -A
git commit -m "feat: Implement DotRenderer with soft anti-aliased circles"
```

---

### Task 5: LineLinkRenderer Node
**Files:**
- `custom_nodes/ys_vision/nodes/line_link_renderer.py`
- `tests/unit/test_line_link_renderer.py`
**Time:** 4-5 hours

#### 5.1 Write Tests First
```python
def test_knn_graph_construction():
    node = LineLinkRendererNode()
    tracks = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])

    layer = node.execute(
        tracks=tracks,
        graph_mode="knn",
        k=2,
        curve="straight",
        width_px=1.0,
        opacity=1.0,
        image_width=100,
        image_height=100
    )[0]

    # Should connect nearest neighbors
    # Check pixels along expected lines
    pass
```

#### 5.2 Implement Line Linking
```python
class LineLinkRendererNode:
    """Connect tracks with lines using various graph strategies"""

    def _build_knn_graph(self, tracks: np.ndarray, k: int) -> List[Tuple[int, int]]:
        """Build k-nearest neighbor graph"""
        from scipy.spatial import KDTree
        tree = KDTree(tracks)
        edges = []
        for i, point in enumerate(tracks):
            distances, indices = tree.query(point, k=k+1)
            for j in indices[1:]:  # Skip self
                if i < j:  # Avoid duplicates
                    edges.append((i, j))
        return edges

    def _draw_line(self, layer: np.ndarray, p1: np.ndarray,
                   p2: np.ndarray, width: float, color: np.ndarray):
        """Draw anti-aliased line"""
        # Use cv2.line with anti-aliasing
        # Or implement Bresenham with AA
        pass
```

#### 5.3 Add Curve Support
```python
def _draw_quadratic_bezier(self, layer: np.ndarray,
                          p0: np.ndarray, p1: np.ndarray,
                          control: np.ndarray, width: float):
    """Draw quadratic Bezier curve"""
    # Sample curve at intervals
    # Connect samples with straight segments
    pass
```

#### 5.4 Commit
```bash
pytest tests/unit/test_line_link_renderer.py -v
git add -A
git commit -m "feat: Add LineLinkRenderer with kNN and radius graph modes"
```

---

### Task 6: LayerMerge Node
**Files:**
- `custom_nodes/ys_vision/nodes/layer_merge.py`
- `tests/unit/test_layer_merge.py`
**Time:** 3 hours

#### 6.1 Write Tests First
```python
def test_blend_modes():
    node = LayerMergeNode()

    # Create test layers
    layer1 = np.ones((100, 100, 4)) * 0.5
    layer2 = np.ones((100, 100, 4)) * 0.3

    # Test normal blend
    merged = node.execute(
        layers=[layer1, layer2],
        blend_modes=["normal", "normal"]
    )[0]

    # Test add blend
    merged_add = node.execute(
        layers=[layer1, layer2],
        blend_modes=["normal", "add"]
    )[0]

    assert merged_add.max() > merged.max()  # Add should be brighter
```

#### 6.2 Implement Blend Modes
```python
class LayerMergeNode:
    """Merge multiple RGBA layers with blend modes"""

    BLEND_MODES = {
        "normal": lambda base, layer: layer,
        "add": lambda base, layer: np.minimum(base + layer, 1.0),
        "screen": lambda base, layer: 1 - (1-base) * (1-layer),
        "multiply": lambda base, layer: base * layer,
        "lighten": lambda base, layer: np.maximum(base, layer)
    }

    def execute(self, layers, blend_modes):
        """Merge layers with specified blend modes"""
        result = layers[0].copy()

        for layer, mode in zip(layers[1:], blend_modes[1:]):
            blend_func = self.BLEND_MODES[mode]
            # Apply blend with alpha compositing
            result = self._alpha_blend(result, layer, blend_func)

        return (result,)
```

#### 6.3 Commit
```bash
pytest tests/unit/test_layer_merge.py -v
git add -A
git commit -m "feat: Implement LayerMerge with multiple blend modes"
```

---

### Task 7: CompositeOver Node
**Files:**
- `custom_nodes/ys_vision/nodes/composite_over.py`
- `tests/unit/test_composite_over.py`
**Time:** 2 hours

#### 7.1 Write Tests First
```python
def test_premultiplied_alpha():
    node = CompositeOverNode()

    # Base image
    image = np.ones((100, 100, 3)) * 0.5

    # Overlay with transparency
    layer = np.zeros((100, 100, 4))
    layer[40:60, 40:60] = [1, 0, 0, 0.5]  # Red with 50% alpha

    result = node.execute(
        image=image,
        layer=layer,
        premultiplied=True
    )[0]

    # Check composite
    assert result[50, 50, 0] > 0.5  # Red channel increased
    assert result[50, 50, 1] < 0.5  # Green channel decreased
```

#### 7.2 Implement Compositing
```python
class CompositeOverNode:
    """Composite RGBA layer over base image"""

    def execute(self, image, layer, premultiplied=True,
                exposure=1.0, soft_clip=False):
        """Apply layer over image with alpha blending"""

        # Ensure correct formats
        image_rgb = ensure_rgb(image)
        layer_rgba = ensure_rgba(layer)

        if not premultiplied:
            # Premultiply alpha
            layer_rgba[:,:,:3] *= layer_rgba[:,:,3:4]

        # Composite formula
        alpha = layer_rgba[:,:,3:4]
        result = image_rgb * (1 - alpha) + layer_rgba[:,:,:3]

        # Apply exposure
        if exposure != 1.0:
            result *= exposure

        # Soft clipping
        if soft_clip:
            result = np.tanh(result)
        else:
            result = np.clip(result, 0, 1)

        return (result,)
```

#### 7.3 Commit
```bash
pytest tests/unit/test_composite_over.py -v
git add -A
git commit -m "feat: Add CompositeOver node with premultiplied alpha support"
```

---

### Task 8: Wire Everything Together
**File:** `custom_nodes/ys_vision/__init__.py`
**Time:** 1 hour

#### 8.1 Update Module Registration
```python
from .nodes.track_detect import TrackDetectNode
from .nodes.palette_map import PaletteMapNode
from .nodes.dot_renderer import DotRendererNode
from .nodes.line_link_renderer import LineLinkRendererNode
from .nodes.layer_merge import LayerMergeNode
from .nodes.composite_over import CompositeOverNode

NODE_CLASS_MAPPINGS = {
    "YSVisionTrackDetect": TrackDetectNode,
    "YSVisionPaletteMap": PaletteMapNode,
    "YSVisionDotRenderer": DotRendererNode,
    "YSVisionLineLinkRenderer": LineLinkRendererNode,
    "YSVisionLayerMerge": LayerMergeNode,
    "YSVisionCompositeOver": CompositeOverNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSVisionTrackDetect": "Track Detect",
    "YSVisionPaletteMap": "Palette Map",
    "YSVisionDotRenderer": "Dot Renderer",
    "YSVisionLineLinkRenderer": "Line Link Renderer",
    "YSVisionLayerMerge": "Layer Merge",
    "YSVisionCompositeOver": "Composite Over"
}
```

#### 8.2 Integration Test
Create `tests/integration/test_mvp_pipeline.py`:
```python
def test_full_mvp_pipeline():
    """Test complete MVP workflow"""
    # Load test image
    # Run through all nodes
    # Verify final output
    pass
```

#### 8.3 Commit
```bash
pytest tests/ -v
git add -A
git commit -m "feat: Complete Phase 1 MVP integration"
```

---

## 🧪 Testing Protocol

### For Each Node:
1. **Unit Tests First** (TDD)
   - Test input/output types
   - Test core functionality
   - Test edge cases

2. **Visual Tests** (Manual)
   - Load in ComfyUI
   - Process sample video
   - Verify visual output

3. **Performance Tests**
   ```python
   import time

   def test_performance():
       start = time.time()
       # Run node 100 times
       elapsed = time.time() - start
       assert elapsed < 2.0  # Should process 100 frames in < 2 seconds
   ```

## 📊 Success Metrics

### Phase 1 is complete when:
- [ ] All 6 nodes implemented and tested
- [ ] Can process 1080p video at 10+ fps (CPU)
- [ ] Test coverage > 80%
- [ ] Visual output matches expected quality
- [ ] No memory leaks over 1000 frames
- [ ] ComfyUI workflow runs without errors

## 🐛 Common Issues & Solutions

### Issue: OpenCV functions not working
**Solution:** Ensure image format is correct (uint8 vs float32)
```python
# OpenCV expects uint8 for many functions
image_uint8 = (image * 255).astype(np.uint8)
```

### Issue: ComfyUI tensor shape mismatch
**Solution:** ComfyUI uses (batch, height, width, channels)
```python
# Add batch dimension if needed
if len(image.shape) == 3:
    image = image[np.newaxis, ...]
```

### Issue: Slow performance
**Solution:** Profile and optimize hot paths
```python
import cProfile
cProfile.run('node.execute(...)')
```

## Next Steps
Continue to `03-PHASE2-EXTENDED.md` for Phase 2 features.