"""
Bounding Box Renderer Node - Phase 2

Renders bounding boxes around tracked points with multiple sizing modes.
Supports fixed sizes, radius-based, and age-based dynamic sizing.

Author: Yambo Studio
Part of: YS-vision-tools Phase 2 (Extended Renderers)
"""

import numpy as np
import cv2
import time
from typing import Dict, Any, List, Tuple

from ..utils import (
    create_rgba_layer,
    numpy_to_comfyui,
    is_gpu_available
)

# Import GPU renderer with fallback
try:
    from ..utils.gpu_rendering import GPUBBoxRenderer
    GPU_RENDERER_AVAILABLE = True
except ImportError:
    GPU_RENDERER_AVAILABLE = False
    GPUBBoxRenderer = None


class BoundingBoxRendererNode:
    """
    Render bounding boxes around tracked points.

    Supports multiple sizing modes:
    - Fixed: All boxes same size
    - From Radius: Based on specified radius
    - From Age: Size grows with track age (logarithmic)

    Features:
    - Adjustable stroke width and fill opacity
    - Rounded corners support
    - Anti-aliased rendering
    - Color per box or unified color
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),
                "box_mode": ([
                    "fixed",
                    "from_radius",
                    "from_age"
                ], {
                    "default": "fixed",
                    "tooltip": "How to determine box size"
                }),
                "stroke_px": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "fill_opacity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Interior fill opacity (0=transparent)"
                }),
                "roundness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Corner roundness (0=square, 1=circular)"
                }),
            },
            "optional": {
                "tracks": ("TRACKS",),
                "boxes": ("BOXES",),  # Pre-computed boxes from object detection
                "ages": ("AGES",),    # Track ages for size calculation
                "palette": ("PALETTE",),

                # Size parameters
                "width": ("INT", {"default": 20, "min": 5, "max": 200, "step": 1}),
                "height": ("INT", {"default": 20, "min": 5, "max": 200, "step": 1}),
                "radius_px": ("FLOAT", {"default": 10.0, "min": 5.0, "max": 100.0, "step": 1.0}),

                # Color
                "color": ("STRING", {"default": "1.0,1.0,1.0"}),  # RGB string

                # GPU acceleration
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (RTX 5090 optimized, 50-100× faster)"
                }),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def __init__(self):
        """Initialize bbox renderer with GPU support"""
        self.gpu_renderer = None
        if GPU_RENDERER_AVAILABLE:
            try:
                self.gpu_renderer = GPUBBoxRenderer(use_gpu=True)
            except Exception as e:
                print(f"[YS-BBOX] GPU renderer init failed: {e}")
                self.gpu_renderer = None

    def execute(self, image_width, image_height, box_mode,
                stroke_px, fill_opacity, roundness, **kwargs):
        """
        Render bounding boxes to RGBA layer.

        Returns:
            LAYER: RGBA layer with rendered boxes
        """

        # DEBUG
        print(f"\n[YS-BBOX] Executing BBoxRenderer")
        tracks = kwargs.get('tracks', np.array([]))
        print(f"[YS-BBOX] tracks type: {type(tracks)}")
        
        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            print(f"[YS-BBOX] BATCH MODE: {len(tracks)} frames")
            batch_layers = []
            
            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                frame_kwargs = kwargs.copy()
                frame_kwargs['tracks'] = frame_tracks
                layer = self._render_single_frame(
                    image_width, image_height, box_mode,
                    stroke_px, fill_opacity, roundness, **frame_kwargs
                )
                print(f"[YS-BBOX] Frame {i} layer shape before append: {layer.shape}")
                batch_layers.append(layer)
                print(f"[YS-BBOX] Frame {i}: {len(frame_tracks)} boxes")
            
            # Stack into batch
            batch_result = np.stack(batch_layers, axis=0)
            print(f"[YS-BBOX] Returning batch: {batch_result.shape}")
            # Don't use numpy_to_comfyui - already in BHWC format, just convert to tensor
            import torch
            return (torch.from_numpy(batch_result.astype(np.float32)),)
        
        # Single frame mode
        print(f"[YS-BBOX] SINGLE MODE")
        layer = self._render_single_frame(
            image_width, image_height, box_mode,
            stroke_px, fill_opacity, roundness, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, image_width, image_height, box_mode,
                             stroke_px, fill_opacity, roundness, **kwargs):
        """Render single frame with GPU/CPU path selection"""

        # Get use_gpu parameter
        use_gpu = kwargs.get('use_gpu', True)

        # Compute box positions and sizes
        boxes = self._compute_boxes(box_mode, **kwargs)

        if len(boxes) == 0:
            # Return empty layer
            return create_rgba_layer(image_height, image_width)

        # GPU path: batch rendering with SDF
        if use_gpu and self.gpu_renderer is not None and len(boxes) > 0:
            try:
                start_time = time.perf_counter()

                # Convert boxes to Nx7 array: [x, y, w, h, r, g, b]
                boxes_array = np.array(boxes, dtype=np.float32)

                # Render with GPU
                layer = self.gpu_renderer.render_boxes_batch(
                    boxes_array,
                    image_width,
                    image_height,
                    stroke_px,
                    fill_opacity,
                    roundness
                )

                gpu_time = (time.perf_counter() - start_time) * 1000
                print(f"[YS-BBOX] GPU rendered {len(boxes)} boxes @ {image_width}x{image_height} in {gpu_time:.2f}ms")

                return layer

            except Exception as e:
                print(f"[YS-BBOX] GPU rendering failed: {e}, falling back to CPU")
                # Fall through to CPU path

        # CPU path: original implementation
        start_time = time.perf_counter()
        layer = self._render_boxes_cpu(
            image_width, image_height, boxes, stroke_px, fill_opacity, roundness
        )
        cpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-BBOX] CPU rendered {len(boxes)} boxes @ {image_width}x{image_height} in {cpu_time:.2f}ms")

        return layer

    def _render_boxes_cpu(self, image_width, image_height, boxes,
                          stroke_px, fill_opacity, roundness):
        """CPU rendering path (original implementation)"""
        # Create empty layer
        layer = create_rgba_layer(image_height, image_width)

        # Render each box
        for box in boxes:
            x, y, w, h = box[:4]
            # Color can be specified per box or use default
            color = np.array(box[4:7]) if len(box) > 4 else np.array([1.0, 1.0, 1.0])

            if roundness > 0:
                self._draw_rounded_rect(layer, x, y, w, h,
                                       roundness, stroke_px,
                                       fill_opacity, color)
            else:
                self._draw_rect(layer, x, y, w, h,
                              stroke_px, fill_opacity, color)

        return layer

    def _compute_boxes(self, mode: str, **kwargs) -> List[np.ndarray]:
        """
        Compute box dimensions based on mode.

        Returns:
            List of boxes, each [x, y, width, height, r, g, b]
        """

        if mode == "fixed":
            # Fixed size for all boxes
            tracks = kwargs.get('tracks', np.array([]))
            if not isinstance(tracks, np.ndarray):
                tracks = np.array(tracks)
            if len(tracks) == 0:
                return []

            width = kwargs.get('width', 20)
            height = kwargs.get('height', 20)
            color = self._parse_color(kwargs.get('color', '1.0,1.0,1.0'))

            boxes = []
            for x, y in tracks:
                # Box centered on point
                boxes.append([
                    x - width / 2, y - height / 2,
                    width, height,
                    *color
                ])
            return boxes

        elif mode == "from_radius":
            # Size based on specified radius
            tracks = kwargs.get('tracks', np.array([]))
            if not isinstance(tracks, np.ndarray):
                tracks = np.array(tracks)
            if len(tracks) == 0:
                return []

            radius = kwargs.get('radius_px', 10.0)
            color = self._parse_color(kwargs.get('color', '1.0,1.0,1.0'))

            boxes = []
            for x, y in tracks:
                # Square box with side = 2 * radius
                size = radius * 2
                boxes.append([
                    x - radius, y - radius,
                    size, size,
                    *color
                ])
            return boxes

        elif mode == "from_age":
            # Size based on track age (logarithmic growth)
            tracks = kwargs.get('tracks', np.array([]))
            if not isinstance(tracks, np.ndarray):
                tracks = np.array(tracks)
            if len(tracks) == 0:
                return []

            ages = kwargs.get('ages', np.ones(len(tracks)))
            if isinstance(ages, list):
                ages = np.array(ages)

            base_size = kwargs.get('radius_px', 10.0)
            color = self._parse_color(kwargs.get('color', '1.0,1.0,1.0'))

            boxes = []
            for (x, y), age in zip(tracks, ages):
                # Logarithmic growth: size increases with age but slows down
                # This creates visual hierarchy without boxes getting too large
                size = base_size * (1 + np.log1p(age) * 0.5)
                size = size * 2  # Convert radius to diameter

                boxes.append([
                    x - size / 2, y - size / 2,
                    size, size,
                    *color
                ])
            return boxes

        return []

    def _draw_rect(self, layer: np.ndarray, x: float, y: float,
                   w: float, h: float, stroke: float,
                   fill_opacity: float, color: np.ndarray):
        """
        Draw rectangle with stroke and optional fill.

        Args:
            layer: RGBA layer to draw on
            x, y: Top-left corner position
            w, h: Width and height
            stroke: Stroke width in pixels
            fill_opacity: Interior fill opacity
            color: RGB color array
        """

        # Convert to integer coordinates
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Clamp to layer bounds
        x1 = max(0, min(x1, layer.shape[1] - 1))
        x2 = max(0, min(x2, layer.shape[1]))
        y1 = max(0, min(y1, layer.shape[0] - 1))
        y2 = max(0, min(y2, layer.shape[0]))

        # Fill interior if requested
        if fill_opacity > 0 and x2 > x1 and y2 > y1:
            # Alpha blend fill
            alpha = fill_opacity
            layer[y1:y2, x1:x2, :3] = \
                layer[y1:y2, x1:x2, :3] * (1 - alpha) + color * alpha
            layer[y1:y2, x1:x2, 3] = np.maximum(
                layer[y1:y2, x1:x2, 3],
                alpha
            )

        # Draw stroke
        if stroke > 0:
            # Create temporary layer for stroke
            temp = np.zeros_like(layer)

            # Draw rectangle outline with anti-aliasing
            rgba = np.concatenate([color, [1.0]])
            cv2.rectangle(
                temp,
                (x1, y1), (x2, y2),
                tuple(rgba),
                int(stroke),
                cv2.LINE_AA
            )

            # Alpha blend stroke
            alpha = temp[:, :, 3:4]
            layer[:, :, :3] = layer[:, :, :3] * (1 - alpha) + temp[:, :, :3] * alpha
            layer[:, :, 3] = np.maximum(layer[:, :, 3], temp[:, :, 3])

    def _draw_rounded_rect(self, layer: np.ndarray, x: float, y: float,
                          w: float, h: float, roundness: float,
                          stroke: float, fill_opacity: float,
                          color: np.ndarray):
        """
        Draw rounded rectangle.

        Args:
            roundness: Corner roundness (0-1), where 1 = circular
        """

        # Calculate corner radius based on roundness
        radius = int(min(w, h) * roundness * 0.5)

        if radius <= 0:
            # Fall back to regular rectangle
            self._draw_rect(layer, x, y, w, h, stroke, fill_opacity, color)
            return

        # Convert to integer coordinates
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Create temporary layer for rendering
        temp = np.zeros_like(layer)
        rgba = np.concatenate([color, [1.0]])

        # Draw filled rounded rectangle if needed
        if fill_opacity > 0:
            # OpenCV doesn't have direct rounded rectangle, so we approximate
            # by drawing rectangle with circles at corners

            # Draw main rectangles
            cv2.rectangle(
                temp,
                (x1 + radius, y1), (x2 - radius, y2),
                tuple(rgba * fill_opacity), -1
            )
            cv2.rectangle(
                temp,
                (x1, y1 + radius), (x2, y2 - radius),
                tuple(rgba * fill_opacity), -1
            )

            # Draw corner circles
            for cx, cy in [
                (x1 + radius, y1 + radius),  # Top-left
                (x2 - radius, y1 + radius),  # Top-right
                (x1 + radius, y2 - radius),  # Bottom-left
                (x2 - radius, y2 - radius),  # Bottom-right
            ]:
                cv2.circle(temp, (cx, cy), radius,
                          tuple(rgba * fill_opacity), -1, cv2.LINE_AA)

        # Draw stroke
        if stroke > 0:
            stroke_temp = np.zeros_like(layer)

            # Draw stroked rounded rectangle
            # Top and bottom edges
            cv2.line(stroke_temp, (x1 + radius, y1), (x2 - radius, y1),
                    tuple(rgba), int(stroke), cv2.LINE_AA)
            cv2.line(stroke_temp, (x1 + radius, y2), (x2 - radius, y2),
                    tuple(rgba), int(stroke), cv2.LINE_AA)

            # Left and right edges
            cv2.line(stroke_temp, (x1, y1 + radius), (x1, y2 - radius),
                    tuple(rgba), int(stroke), cv2.LINE_AA)
            cv2.line(stroke_temp, (x2, y1 + radius), (x2, y2 - radius),
                    tuple(rgba), int(stroke), cv2.LINE_AA)

            # Corner arcs
            for cx, cy, start, end in [
                (x1 + radius, y1 + radius, 180, 270),  # Top-left
                (x2 - radius, y1 + radius, 270, 360),  # Top-right
                (x1 + radius, y2 - radius, 90, 180),   # Bottom-left
                (x2 - radius, y2 - radius, 0, 90),     # Bottom-right
            ]:
                cv2.ellipse(
                    stroke_temp, (cx, cy), (radius, radius),
                    0, start, end, tuple(rgba), int(stroke), cv2.LINE_AA
                )

            # Blend stroke
            alpha = stroke_temp[:, :, 3:4]
            temp[:, :, :3] = temp[:, :, :3] * (1 - alpha) + stroke_temp[:, :, :3] * alpha
            temp[:, :, 3] = np.maximum(temp[:, :, 3], stroke_temp[:, :, 3])

        # Blend final result with layer
        alpha = temp[:, :, 3:4]
        layer[:, :, :3] = layer[:, :, :3] * (1 - alpha) + temp[:, :, :3] * alpha
        layer[:, :, 3] = np.maximum(layer[:, :, 3], temp[:, :, 3])

    def _parse_color(self, color_str: str) -> np.ndarray:
        """Parse color string to RGB array."""
        try:
            return np.array([float(x.strip()) for x in color_str.split(',')])[:3]
        except:
            return np.array([1.0, 1.0, 1.0])


# Node registration
NODE_CLASS_MAPPINGS = {"YS_BBoxRenderer": BoundingBoxRendererNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_BBoxRenderer": "Bounding Box Renderer 📦"}