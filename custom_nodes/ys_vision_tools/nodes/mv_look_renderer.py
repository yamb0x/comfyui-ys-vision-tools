"""
Machine Vision Look Renderer Node - Phase 2

Apply machine vision / CRT / surveillance camera aesthetic effects.
Creates retro technical/surveillance looks with scanlines, chromatic aberration, etc.

Author: Yambo Studio
Part of: YS-vision-tools Phase 2 (Extended Renderers)
"""

import numpy as np
import torch
from scipy.ndimage import shift
from typing import Dict, Any, Tuple, Optional

from ..utils import (
    comfyui_to_numpy,
    numpy_to_comfyui,
    create_rgba_layer
)


class MVLookRendererNode:
    """
    Apply machine vision aesthetic effects.

    Effects include:
    - CRT scanlines
    - Chromatic aberration (lens distortion)
    - Vignette darkening
    - Film grain / sensor noise
    - Color tinting (green/cyan surveillance look)

    Can output as modified image or as overlay layer.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "scanline_intensity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "CRT scanline darkness"
                }),
                "chroma_offset_px": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Chromatic aberration offset"
                }),
                "vignette": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Edge darkening strength"
                }),
                "noise": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Film grain / sensor noise"
                }),
                "as_layer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output as overlay layer (True) or modified image (False)"
                }),
            },
            "optional": {
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scanline_spacing": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Pixels between scanlines"
                }),
                "color_tint": ("STRING", {
                    "default": "0.0,1.0,0.8",
                    "tooltip": "RGB tint (surveillance green/cyan)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "LAYER")
    RETURN_NAMES = ("image", "layer")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Effects"

    def execute(self, image: torch.Tensor, scanline_intensity: float,
                chroma_offset_px: float, vignette: float, noise: float,
                as_layer: bool, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply machine vision look effects.

        Returns:
            image: Modified image (original if as_layer=True)
            layer: Effect layer (None if as_layer=False)
        """

        # Convert to numpy
        image_np = comfyui_to_numpy(image)

        # Handle batch (take first frame)
        if len(image_np.shape) == 4:
            original_batch = image_np
            image_np = image_np[0]
        else:
            original_batch = None

        # Apply effects in sequence
        result = image_np.copy()

        if scanline_intensity > 0:
            result = self._apply_scanlines(
                result,
                scanline_intensity,
                kwargs.get('scanline_spacing', 2)
            )

        if chroma_offset_px > 0:
            result = self._apply_chromatic_aberration(result, chroma_offset_px)

        if vignette > 0:
            result = self._apply_vignette(result, vignette)

        if noise > 0:
            result = self._apply_noise(result, noise)

        # Apply color tint
        tint_str = kwargs.get('color_tint', '0.0,1.0,0.8')
        tint = self._parse_color(tint_str)
        if not np.allclose(tint, [1.0, 1.0, 1.0]):
            result = self._apply_color_tint(result, tint)

        if as_layer:
            # Return as RGBA layer
            opacity = kwargs.get('opacity', 1.0)
            h, w = result.shape[:2]
            layer = create_rgba_layer(h, w)
            layer[:, :, :3] = result
            layer[:, :, 3] = opacity

            # Return original image + layer
            return (image, numpy_to_comfyui(layer))
        else:
            # Return modified image
            # Restore batch dimension if needed
            if original_batch is not None:
                result_batch = original_batch.copy()
                result_batch[0] = result
                result = result_batch

            return (numpy_to_comfyui(result), None)

    def _apply_scanlines(self, image: np.ndarray, intensity: float,
                        spacing: int) -> np.ndarray:
        """
        Add CRT-style scanlines.

        Args:
            intensity: Darkening strength
            spacing: Pixels between scanlines
        """

        result = image.copy()
        h = image.shape[0]

        for y in range(0, h, spacing):
            # Darken every nth line
            result[y, :] *= (1.0 - intensity)

            # Optional: Add subtle bright line after dark (CRT phosphor glow)
            if y + 1 < h and spacing > 1:
                result[y + 1, :] *= (1.0 + intensity * 0.2)
                result[y + 1, :] = np.clip(result[y + 1, :], 0, 1)

        return result

    def _apply_chromatic_aberration(self, image: np.ndarray,
                                   offset: float) -> np.ndarray:
        """
        Simulate lens chromatic aberration.

        Shifts red channel left, blue channel right.
        Creates color fringing at edges.
        """

        result = image.copy()

        if image.shape[2] >= 3:
            # Shift red channel left
            result[:, :, 0] = shift(image[:, :, 0], [0, -offset], mode='nearest')

            # Keep green centered
            # result[:, :, 1] = image[:, :, 1]

            # Shift blue channel right
            result[:, :, 2] = shift(image[:, :, 2], [0, offset], mode='nearest')

        return result

    def _apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Add vignette darkening at edges.

        Args:
            strength: Vignette strength (0-1)
        """

        h, w = image.shape[:2]
        result = image.copy()

        # Create radial gradient
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]

        # Distance from center (normalized)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_dist

        # Smooth falloff (quadratic)
        vignette_mask = 1.0 - (dist**2 * strength)
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # Apply to all channels
        for i in range(result.shape[2]):
            result[:, :, i] *= vignette_mask

        return result

    def _apply_noise(self, image: np.ndarray, amount: float) -> np.ndarray:
        """
        Add film grain / sensor noise.

        Args:
            amount: Noise strength (0-0.5 typical)
        """

        # Generate gaussian noise
        noise = np.random.normal(0, amount, image.shape)

        # Add noise and clip
        result = image + noise
        result = np.clip(result, 0, 1)

        return result

    def _apply_color_tint(self, image: np.ndarray, tint: np.ndarray) -> np.ndarray:
        """
        Apply color tint/filter.

        Args:
            tint: RGB color multiplier
        """

        result = image.copy()

        for i in range(min(3, len(tint))):
            result[:, :, i] *= tint[i]

        return np.clip(result, 0, 1)

    def _parse_color(self, color_str: str) -> np.ndarray:
        """Parse color string to RGB array."""
        try:
            return np.array([float(x.strip()) for x in color_str.split(',')])[:3]
        except:
            return np.array([1.0, 1.0, 1.0])


# Node registration
NODE_CLASS_MAPPINGS = {"YS_MVLookRenderer": MVLookRendererNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_MVLookRenderer": "Machine Vision Look 📹"}
