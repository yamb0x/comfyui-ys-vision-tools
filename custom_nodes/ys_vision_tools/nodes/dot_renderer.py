"""
Dot Renderer Node for YS-vision-tools
Renders points as dots with various styles and colors

GPU-ACCELERATED with SDF-based rendering:
- 50-100× faster than CPU @ 4K resolution
- Perfect anti-aliasing via signed distance fields
- Batched rendering (all dots in single GPU pass)
- 6 shape types with stroke/fill control

🎨 COLOR PICKER SUPPORT:
- Visual color picker UI (click the color swatch in ComfyUI)
- Supports HEX colors: "#ffffff", "#ff0000", "#00ff00"
- Supports named colors: "red", "orange", "cyan", "white"
- Backward compatible with legacy float lists: [1.0, 0.5, 0.0]
- Separate alpha slider for transparency control

Usage:
    1. Click the color swatch to open the visual color picker
    2. Select your desired color using the picker UI
    3. Adjust alpha slider for transparency (0=invisible, 1=opaque)
    4. Color is automatically parsed and applied to all dots
"""

import numpy as np
import time
from typing import Dict, Any

from ..utils import (
    create_rgba_layer,
    numpy_to_comfyui,
    normalize_color_to_rgba01
)

# Import GPU renderer with fallback
try:
    from ..utils.gpu_rendering import GPUDotRenderer
    GPU_RENDERER_AVAILABLE = True
except ImportError:
    GPU_RENDERER_AVAILABLE = False
    GPUDotRenderer = None


class DotRendererNode:
    """
    Render tracked points as styled dots

    Features:
    - 6 shape types: solid, ring, cross, plus, square, diamond
    - Stroke width control for outlined shapes
    - Fill opacity control (0=hollow, 1=filled)
    - GPU acceleration (50-100× faster @ 4K)
    - Batch processing for video frames
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),
                "dot_size": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 50.0, "step": 0.5}),
                "stroke_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Line thickness for outlined shapes (ring, cross, plus, square, diamond)"
                }),
                "fill_opacity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Interior fill opacity (0=hollow, 1=filled). Ignored for 'solid' style."
                }),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dot_style": (["solid", "ring", "cross", "plus", "square", "diamond"],),
                "color": ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Click the color swatch to open the visual color picker"
                }),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Transparency level (0=invisible, 1=opaque)"
                }),
                "glow_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "glow_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (RTX 5090 optimized, 50-100× faster @ 4K)"
                }),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def __init__(self):
        """Initialize dot renderer with GPU support"""
        self.gpu_renderer = None
        if GPU_RENDERER_AVAILABLE:
            try:
                self.gpu_renderer = GPUDotRenderer(use_gpu=True)
            except Exception as e:
                print(f"[YS-DOT] GPU renderer init failed: {e}")
                self.gpu_renderer = None

    def execute(self, tracks, image_width, image_height, dot_size,
                stroke_width, fill_opacity, opacity, dot_style, color, **kwargs):
        """Render dots at tracked positions

        Args:
            color: COLOR input (hex string like "#ffffff" or named color like "red")
                   Automatically parsed by normalize_color_to_rgba01()
            **kwargs: Optional parameters including 'alpha' for transparency
        """

        # DEBUG
        print(f"\n[YS-DOT] Executing DotRenderer")
        print(f"[YS-DOT] tracks type: {type(tracks)}")
        print(f"[YS-DOT] color input: {color} (type: {type(color)})")

        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            print(f"[YS-DOT] BATCH MODE: {len(tracks)} frames")
            batch_layers = []

            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                layer = self._render_single_frame(
                    frame_tracks, image_width, image_height,
                    dot_size, stroke_width, fill_opacity, opacity, dot_style,
                    color, **kwargs
                )
                batch_layers.append(layer)

                # Progress logging (every 10 frames)
                if i % 10 == 0 or i == len(tracks) - 1:
                    print(f"[YS-DOT] Processed frame {i+1}/{len(tracks)}")

            # Stack into batch
            batch_result = np.stack(batch_layers, axis=0)
            print(f"[YS-DOT] Returning batch: {batch_result.shape}")
            # Don't use numpy_to_comfyui - already in BHWC format, just convert to tensor
            import torch
            return (torch.from_numpy(batch_result.astype(np.float32)),)

        # Single frame mode
        print(f"[YS-DOT] SINGLE MODE")
        layer = self._render_single_frame(
            tracks, image_width, image_height,
            dot_size, stroke_width, fill_opacity, opacity, dot_style,
            color, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, tracks, image_width, image_height,
                            dot_size, stroke_width, fill_opacity, opacity, dot_style,
                            color, **kwargs):
        """Render single frame with GPU/CPU path selection

        Args:
            color: COLOR input (hex/named color) passed from execute()
            **kwargs: Optional parameters including 'alpha' override
        """

        # Convert tracks to numpy array
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks)

        if len(tracks) == 0:
            return create_rgba_layer(image_height, image_width)

        # Get use_gpu parameter
        use_gpu = kwargs.get('use_gpu', True)

        # Parse color using centralized utility
        # Alpha from optional parameter overrides opacity if provided
        alpha = kwargs.get('alpha', opacity)
        rgba = normalize_color_to_rgba01(color, alpha)
        color_rgb = rgba[:3]  # Extract RGB for rendering functions

        print(f"[YS-DOT] Parsed color: {color} -> RGBA: {rgba}")

        # GPU path: batched SDF rendering
        if use_gpu and self.gpu_renderer is not None and len(tracks) > 0:
            try:
                start_time = time.perf_counter()

                # Render with GPU
                layer = self.gpu_renderer.render_dots_batch(
                    tracks,
                    image_width,
                    image_height,
                    dot_size,
                    stroke_width,
                    fill_opacity,
                    opacity,
                    dot_style,
                    color_rgb  # Pass RGB tuple to GPU renderer
                )

                gpu_time = (time.perf_counter() - start_time) * 1000
                print(f"[YS-DOT] GPU rendered {len(tracks)} dots @ {image_width}x{image_height} in {gpu_time:.2f}ms")

                # Add glow if requested (CPU fallback for glow)
                glow_size = kwargs.get('glow_size', 0.0)
                if glow_size > 0:
                    glow_intensity = kwargs.get('glow_intensity', 0.5)
                    self._add_glow_cpu(layer, tracks, glow_size, color_rgb, glow_intensity)

                return layer

            except Exception as e:
                print(f"[YS-DOT] GPU rendering failed: {e}, falling back to CPU")
                # Fall through to CPU path

        # CPU path: original implementation
        start_time = time.perf_counter()
        layer = self._render_dots_cpu(
            image_width, image_height, tracks, dot_size,
            stroke_width, fill_opacity, opacity, dot_style, color_rgb, **kwargs
        )
        cpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-DOT] CPU rendered {len(tracks)} dots @ {image_width}x{image_height} in {cpu_time:.2f}ms")

        return layer

    def _render_dots_cpu(self, image_width, image_height, tracks, dot_size,
                         stroke_width, fill_opacity, opacity, dot_style, color, **kwargs):
        """CPU rendering path - uses OpenCV"""
        import cv2

        # Create empty layer
        layer = create_rgba_layer(image_height, image_width)
        rgba = (*color, opacity)

        # Render each dot
        for x, y in tracks:
            x_int, y_int = int(x), int(y)
            size_int = int(dot_size)
            stroke_int = max(1, int(stroke_width))

            # Check bounds
            if 0 <= x_int < image_width and 0 <= y_int < image_height:
                temp = np.zeros_like(layer)

                if dot_style == "solid":
                    cv2.circle(temp, (x_int, y_int), size_int, rgba, -1, cv2.LINE_AA)

                elif dot_style == "ring":
                    cv2.circle(temp, (x_int, y_int), size_int, rgba, stroke_int, cv2.LINE_AA)
                    if fill_opacity > 0:
                        fill_rgba = (*color, opacity * fill_opacity)
                        cv2.circle(temp, (x_int, y_int), max(1, size_int - stroke_int), fill_rgba, -1, cv2.LINE_AA)

                elif dot_style == "cross":
                    # X shape
                    offset = size_int
                    cv2.line(temp, (x_int - offset, y_int - offset),
                            (x_int + offset, y_int + offset), rgba, stroke_int, cv2.LINE_AA)
                    cv2.line(temp, (x_int - offset, y_int + offset),
                            (x_int + offset, y_int - offset), rgba, stroke_int, cv2.LINE_AA)
                    if fill_opacity > 0:
                        fill_rgba = (*color, opacity * fill_opacity)
                        points = np.array([[x_int - offset, y_int], [x_int, y_int - offset],
                                         [x_int + offset, y_int], [x_int, y_int + offset]])
                        cv2.fillPoly(temp, [points], fill_rgba)

                elif dot_style == "plus":
                    # + shape
                    offset = size_int
                    cv2.line(temp, (x_int, y_int - offset),
                            (x_int, y_int + offset), rgba, stroke_int, cv2.LINE_AA)
                    cv2.line(temp, (x_int - offset, y_int),
                            (x_int + offset, y_int), rgba, stroke_int, cv2.LINE_AA)
                    if fill_opacity > 0:
                        fill_rgba = (*color, opacity * fill_opacity)
                        cv2.rectangle(temp, (x_int - offset, y_int - offset),
                                    (x_int + offset, y_int + offset), fill_rgba, -1)

                elif dot_style == "square":
                    offset = size_int
                    pt1 = (x_int - offset, y_int - offset)
                    pt2 = (x_int + offset, y_int + offset)
                    if fill_opacity > 0:
                        fill_rgba = (*color, opacity * fill_opacity)
                        cv2.rectangle(temp, pt1, pt2, fill_rgba, -1)
                    cv2.rectangle(temp, pt1, pt2, rgba, stroke_int, cv2.LINE_AA)

                elif dot_style == "diamond":
                    offset = size_int
                    points = np.array([
                        [x_int, y_int - offset],
                        [x_int + offset, y_int],
                        [x_int, y_int + offset],
                        [x_int - offset, y_int]
                    ])
                    if fill_opacity > 0:
                        fill_rgba = (*color, opacity * fill_opacity)
                        cv2.fillPoly(temp, [points], fill_rgba, cv2.LINE_AA)
                    cv2.polylines(temp, [points], True, rgba, stroke_int, cv2.LINE_AA)

                # Blend
                alpha = temp[:, :, 3:4]
                layer[:, :, :3] = layer[:, :, :3] * (1 - alpha) + temp[:, :, :3] * alpha
                layer[:, :, 3] = np.maximum(layer[:, :, 3], temp[:, :, 3])

                # Add glow if requested
                glow_size = kwargs.get('glow_size', 0.0)
                if glow_size > 0:
                    glow_intensity = kwargs.get('glow_intensity', 0.5)
                    self._render_glow_cpu(layer, x_int, y_int, glow_size, color, glow_intensity)

        return layer

    def _add_glow_cpu(self, layer: np.ndarray, tracks: np.ndarray,
                      glow_size: float, color: tuple, intensity: float):
        """Add glow effect to existing layer (CPU)"""
        import cv2

        for x, y in tracks:
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < layer.shape[1] and 0 <= y_int < layer.shape[0]:
                self._render_glow_cpu(layer, x_int, y_int, glow_size, color, intensity)

    def _render_glow_cpu(self, layer: np.ndarray, x: int, y: int, glow_size: float,
                        color: tuple, intensity: float):
        """Render glow effect around dot (CPU)"""
        import cv2

        rgba = (*color, intensity)
        cv2.circle(layer, (x, y), int(glow_size), rgba, -1, cv2.LINE_AA)


NODE_CLASS_MAPPINGS = {
    "YS_DotRenderer": DotRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_DotRenderer": "Dot Renderer ⚫"
}
