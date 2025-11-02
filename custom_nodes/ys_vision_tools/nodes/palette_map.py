"""
Palette Map Node for YS-vision-tools
Creates and manages color palettes for rendering
"""

import numpy as np
from typing import Dict, Any, List

from ..utils import numpy_to_comfyui


class PaletteMapNode:
    """Create color palettes for visualization"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "palette_type": ([
                    "rainbow",
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cool",
                    "warm",
                    "custom_gradient",
                ],),
                "num_colors": ("INT", {"default": 256, "min": 2, "max": 1024, "step": 1}),
            },
            "optional": {
                "custom_color_1": ("STRING", {"default": "1.0,0.0,0.0"}),  # Red
                "custom_color_2": ("STRING", {"default": "0.0,1.0,0.0"}),  # Green
                "custom_color_3": ("STRING", {"default": "0.0,0.0,1.0"}),  # Blue
            }
        }

    RETURN_TYPES = ("PALETTE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("palette", "color_1", "color_2", "color_3", "color_4", "color_5")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Utilities"

    def execute(self, palette_type, num_colors, **kwargs):
        """
        Generate color palette and extract individual colors.

        Returns:
            - palette: Full color palette array
            - color_1 to color_5: Individual colors evenly distributed across the palette
                                  (as RGB strings for node inputs)

        The 5 individual colors are extracted at evenly-spaced positions:
        - color_1: start of palette (0%)
        - color_2: 25% through palette
        - color_3: 50% through palette (middle)
        - color_4: 75% through palette
        - color_5: end of palette (100%)
        """

        if palette_type == "rainbow":
            palette = self._rainbow_palette(num_colors)
        elif palette_type == "viridis":
            palette = self._viridis_palette(num_colors)
        elif palette_type == "plasma":
            palette = self._plasma_palette(num_colors)
        elif palette_type == "inferno":
            palette = self._inferno_palette(num_colors)
        elif palette_type == "magma":
            palette = self._magma_palette(num_colors)
        elif palette_type == "cool":
            palette = self._cool_palette(num_colors)
        elif palette_type == "warm":
            palette = self._warm_palette(num_colors)
        elif palette_type == "custom_gradient":
            palette = self._custom_gradient(num_colors, **kwargs)
        else:
            # Default to rainbow
            palette = self._rainbow_palette(num_colors)

        # Extract 5 evenly-spaced colors from the palette
        # This allows one palette to feed multiple nodes with harmonious colors
        indices = np.linspace(0, num_colors - 1, 5, dtype=int)

        individual_colors = []
        for idx in indices:
            color_rgb = palette[idx]  # (R, G, B) in [0, 1]
            # Convert to string format for ComfyUI node inputs
            color_str = f"{color_rgb[0]:.3f},{color_rgb[1]:.3f},{color_rgb[2]:.3f}"
            individual_colors.append(color_str)

        return (
            palette,              # Full palette array
            individual_colors[0], # Color 1 (start - 0%)
            individual_colors[1], # Color 2 (25%)
            individual_colors[2], # Color 3 (middle - 50%)
            individual_colors[3], # Color 4 (75%)
            individual_colors[4], # Color 5 (end - 100%)
        )

    def _rainbow_palette(self, n: int) -> np.ndarray:
        """Generate rainbow color palette"""
        hues = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        for i, h in enumerate(hues):
            # HSV to RGB conversion
            colors[i] = self._hsv_to_rgb(h, 1.0, 1.0)

        return colors

    def _viridis_palette(self, n: int) -> np.ndarray:
        """Viridis color palette (perceptually uniform)"""
        # Simplified viridis approximation
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = 0.267 + 0.105 * t  # R
        colors[:, 1] = 0.005 + 0.566 * t  # G
        colors[:, 2] = 0.329 + 0.428 * t  # B

        return np.clip(colors, 0, 1)

    def _plasma_palette(self, n: int) -> np.ndarray:
        """Plasma color palette"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = 0.050 + 0.900 * t  # R
        colors[:, 1] = 0.030 + 0.500 * t**2  # G
        colors[:, 2] = 0.900 - 0.850 * t  # B

        return np.clip(colors, 0, 1)

    def _inferno_palette(self, n: int) -> np.ndarray:
        """Inferno color palette"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = np.minimum(1.0, 1.5 * t)  # R
        colors[:, 1] = np.maximum(0, 1.5 * (t - 0.4))  # G
        colors[:, 2] = np.maximum(0, 4 * (t - 0.75))  # B

        return np.clip(colors, 0, 1)

    def _magma_palette(self, n: int) -> np.ndarray:
        """Magma color palette"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = np.minimum(1.0, 1.3 * t)  # R
        colors[:, 1] = 0.15 * t + 0.85 * t**2  # G
        colors[:, 2] = 0.3 + 0.5 * np.sin(np.pi * t)  # B

        return np.clip(colors, 0, 1)

    def _cool_palette(self, n: int) -> np.ndarray:
        """Cool color palette (cyan to blue)"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = 0.0  # R
        colors[:, 1] = 1.0 - t  # G
        colors[:, 2] = 1.0  # B

        return colors

    def _warm_palette(self, n: int) -> np.ndarray:
        """Warm color palette (yellow to red)"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))

        colors[:, 0] = 1.0  # R
        colors[:, 1] = 1.0 - t  # G
        colors[:, 2] = 0.0  # B

        return colors

    def _custom_gradient(self, n: int, **kwargs) -> np.ndarray:
        """Custom gradient from provided colors"""

        # Parse custom colors
        colors_list = []
        for i in range(1, 4):
            color_str = kwargs.get(f'custom_color_{i}', None)
            if color_str:
                try:
                    color = np.array([float(x.strip()) for x in color_str.split(',')])[:3]
                    colors_list.append(color)
                except:
                    pass

        if len(colors_list) < 2:
            # Fallback to rainbow
            return self._rainbow_palette(n)

        # Interpolate between colors
        palette = np.zeros((n, 3))
        segments = len(colors_list) - 1

        for i in range(n):
            # Determine which segment
            t = i / (n - 1)
            segment_idx = min(int(t * segments), segments - 1)
            segment_t = (t * segments) - segment_idx

            # Interpolate
            c1 = colors_list[segment_idx]
            c2 = colors_list[segment_idx + 1]
            palette[i] = c1 * (1 - segment_t) + c2 * segment_t

        return palette

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """Convert HSV to RGB"""
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i = i % 6

        if i == 0:
            return np.array([v, t, p])
        elif i == 1:
            return np.array([q, v, p])
        elif i == 2:
            return np.array([p, v, t])
        elif i == 3:
            return np.array([p, q, v])
        elif i == 4:
            return np.array([t, p, v])
        else:
            return np.array([v, p, q])


NODE_CLASS_MAPPINGS = {
    "YS_PaletteMap": PaletteMapNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_PaletteMap": "Palette Map 🎨"
}
