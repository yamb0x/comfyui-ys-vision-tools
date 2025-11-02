"""
Dot Renderer Node for YS-vision-tools
Renders points as dots with various styles and colors
"""

import numpy as np
import cv2
from typing import Dict, Any

from ..utils import (
    create_rgba_layer,
    numpy_to_comfyui
)


class DotRendererNode:
    """Render tracked points as styled dots"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),
                "dot_size": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 50.0, "step": 0.5}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dot_style": (["solid", "ring", "cross", "plus", "square", "diamond"],),
            },
            "optional": {
                "color": ("STRING", {"default": "1.0,1.0,1.0"}),  # RGB as string
                "glow_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "glow_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def execute(self, tracks, image_width, image_height, dot_size, opacity, dot_style, **kwargs):
        """Render dots at tracked positions"""

        # DEBUG
        print(f"\n[YS-DOT] Executing DotRenderer")
        print(f"[YS-DOT] tracks type: {type(tracks)}")
        
        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            print(f"[YS-DOT] BATCH MODE: {len(tracks)} frames")
            batch_layers = []
            
            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                layer = self._render_single_frame(
                    frame_tracks, image_width, image_height, 
                    dot_size, opacity, dot_style, **kwargs
                )
                batch_layers.append(layer)
                print(f"[YS-DOT] Frame {i}: {len(frame_tracks)} dots")
            
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
            dot_size, opacity, dot_style, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, tracks, image_width, image_height, 
                            dot_size, opacity, dot_style, **kwargs):
        """Render single frame - extracted to avoid duplication"""
        
        # Convert tracks to numpy array
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks)

        # Create empty layer
        layer = create_rgba_layer(image_height, image_width)

        if len(tracks) == 0:
            return layer

        # Parse color
        color_str = kwargs.get('color', '1.0,1.0,1.0')
        try:
            color = np.array([float(x.strip()) for x in color_str.split(',')])[:3]
        except:
            color = np.array([1.0, 1.0, 1.0])

        # Render each dot
        for x, y in tracks:
            x_int, y_int = int(x), int(y)

            # Check bounds
            if 0 <= x_int < image_width and 0 <= y_int < image_height:
                self._render_dot(layer, x_int, y_int, dot_size, color, opacity, dot_style)

                # Add glow if requested
                glow_size = kwargs.get('glow_size', 0.0)
                if glow_size > 0:
                    glow_intensity = kwargs.get('glow_intensity', 0.5)
                    self._render_glow(layer, x_int, y_int, glow_size, color, glow_intensity)

        return layer

    def _render_dot(self, layer: np.ndarray, x: int, y: int, size: float,
                   color: np.ndarray, opacity: float, style: str):
        """Render a single dot with specified style"""

        rgba = np.concatenate([color, [opacity]])

        if style == "solid":
            cv2.circle(layer, (x, y), int(size), tuple(rgba), -1, cv2.LINE_AA)

        elif style == "ring":
            cv2.circle(layer, (x, y), int(size), tuple(rgba), max(1, int(size / 3)), cv2.LINE_AA)

        elif style == "cross":
            # Draw X shape
            offset = int(size)
            cv2.line(layer, (x - offset, y - offset), (x + offset, y + offset), tuple(rgba), 2, cv2.LINE_AA)
            cv2.line(layer, (x - offset, y + offset), (x + offset, y - offset), tuple(rgba), 2, cv2.LINE_AA)

        elif style == "plus":
            # Draw + shape
            offset = int(size)
            cv2.line(layer, (x, y - offset), (x, y + offset), tuple(rgba), 2, cv2.LINE_AA)
            cv2.line(layer, (x - offset, y), (x + offset, y), tuple(rgba), 2, cv2.LINE_AA)

        elif style == "square":
            offset = int(size)
            pt1 = (x - offset, y - offset)
            pt2 = (x + offset, y + offset)
            cv2.rectangle(layer, pt1, pt2, tuple(rgba), -1, cv2.LINE_AA)

        elif style == "diamond":
            offset = int(size)
            points = np.array([
                [x, y - offset],
                [x + offset, y],
                [x, y + offset],
                [x - offset, y]
            ])
            cv2.fillPoly(layer, [points], tuple(rgba), cv2.LINE_AA)

    def _render_glow(self, layer: np.ndarray, x: int, y: int, glow_size: float,
                    color: np.ndarray, intensity: float):
        """Render glow effect around dot"""

        rgba = np.concatenate([color, [intensity]])
        cv2.circle(layer, (x, y), int(glow_size), tuple(rgba), -1, cv2.LINE_AA)


NODE_CLASS_MAPPINGS = {
    "YS_DotRenderer": DotRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_DotRenderer": "Dot Renderer ⚫"
}