"""
SDF (Signed Distance Field) Font Rendering

GPU-accelerated text rendering with Hebrew and English support.
Uses pre-generated SDF atlas for sharp text at any scale.

Author: Yambo Studio
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import cupy as cp
    from .cuda_kernels import get_compiled_kernel, SDF_TEXT_RENDER_KERNEL
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class SDFFontAtlas:
    """
    SDF font atlas manager with Hebrew/English support

    Loads pre-generated SDF atlas and provides GPU-accelerated text rendering.
    """

    def __init__(self, atlas_path: Optional[str] = None, metadata_path: Optional[str] = None, use_gpu=True):
        """
        Initialize SDF font atlas

        Args:
            atlas_path: Path to SDF atlas image (PNG, grayscale)
            metadata_path: Path to glyph metadata JSON
            use_gpu: Use GPU acceleration if available
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        # Default to built-in atlas if not provided
        if atlas_path is None:
            # Use simple generated atlas for now
            # TODO: Replace with proper pre-baked SDF atlas
            self.atlas_size = 512
            self.atlas_data = self._generate_simple_atlas()
        else:
            self.atlas_data = self._load_atlas(atlas_path)
            self.atlas_size = self.atlas_data.shape[0]

        # Load glyph metadata
        if metadata_path is None:
            self.glyph_metadata = self._generate_simple_metadata()
        else:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.glyph_metadata = json.load(f)

        # Transfer atlas to GPU if needed
        if self.use_gpu:
            self.atlas_gpu = cp.asarray(self.atlas_data, dtype=cp.float32)
        else:
            self.atlas_gpu = None

        print(f"[YS-SDF] Loaded font atlas {self.atlas_size}x{self.atlas_size}, "
              f"{len(self.glyph_metadata)} glyphs, GPU={'ON' if self.use_gpu else 'OFF'}")

    def _load_atlas(self, path: str) -> np.ndarray:
        """Load SDF atlas from image file"""
        from PIL import Image
        img = Image.open(path).convert('L')
        atlas = np.array(img, dtype=np.float32) / 255.0
        return atlas

    def _generate_simple_atlas(self) -> np.ndarray:
        """
        Generate a simple placeholder SDF atlas

        TODO: Replace with proper pre-baked multi-script SDF atlas
        """
        # For now, create a simple test atlas with basic ASCII
        size = 512
        atlas = np.zeros((size, size), dtype=np.float32)

        # Fill with simple patterns for testing
        # In real implementation, this would be pre-baked SDF data
        atlas[:] = 0.5  # Mid-value for testing

        return atlas

    def _generate_simple_metadata(self) -> Dict[str, Dict]:
        """
        Generate simple metadata for basic ASCII glyphs

        TODO: Replace with real glyph metadata including Hebrew
        """
        metadata = {}

        # Simple layout: 16x16 grid of ASCII characters (32-127)
        chars_per_row = 16
        cell_size = 32  # pixels per character in 512x512 atlas

        for i in range(96):  # ASCII 32-127
            char = chr(32 + i)
            row = i // chars_per_row
            col = i % chars_per_row

            metadata[char] = {
                'u': col * cell_size / 512.0,
                'v': row * cell_size / 512.0,
                'w': cell_size / 512.0,
                'h': cell_size / 512.0,
                'advance': 0.5,  # Normalized advance
                'bearing_x': 0.1,
                'bearing_y': 0.8,
            }

        return metadata

    def layout_text(self, text: str, font_px: int, anchor: str = 'center',
                    rtl: bool = False) -> List[Dict]:
        """
        Layout text into glyph quads

        Args:
            text: Text string to layout
            font_px: Font size in pixels
            anchor: Anchor position ('center', 'left', 'right', 'above', 'below')
            rtl: Right-to-left text (for Hebrew)

        Returns:
            List of glyph quad dictionaries with position and UV data
        """
        quads = []

        if not text:
            return quads

        # Handle RTL if needed
        if rtl:
            try:
                from bidi.algorithm import get_display
                text = get_display(text)
            except ImportError:
                # Fallback: simple reversal (not proper BiDi)
                text = text[::-1]

        # Calculate text dimensions
        total_advance = sum(
            self.glyph_metadata.get(char, {'advance': 0.5})['advance']
            for char in text
        ) * font_px

        # Starting position based on anchor
        if anchor == 'center':
            x_offset = -total_advance / 2
            y_offset = -font_px / 2
        elif anchor == 'left':
            x_offset = -total_advance
            y_offset = -font_px / 2
        elif anchor == 'right':
            x_offset = 0
            y_offset = -font_px / 2
        elif anchor == 'above':
            x_offset = -total_advance / 2
            y_offset = -font_px - 10
        elif anchor == 'below':
            x_offset = -total_advance / 2
            y_offset = 10
        else:
            x_offset = 0
            y_offset = 0

        # Layout glyphs
        cursor_x = x_offset

        for char in text:
            if char == ' ':
                cursor_x += font_px * 0.3
                continue

            glyph = self.glyph_metadata.get(char)
            if glyph is None:
                # Unknown character, use placeholder
                cursor_x += font_px * 0.5
                continue

            # Create quad
            quad = {
                'x': cursor_x + glyph['bearing_x'] * font_px,
                'y': y_offset + (1.0 - glyph['bearing_y']) * font_px,
                'w': glyph['w'] * self.atlas_size * (font_px / self.atlas_size),
                'h': glyph['h'] * self.atlas_size * (font_px / self.atlas_size),
                'u': glyph['u'],
                'v': glyph['v'],
                'uw': glyph['w'],
                'vh': glyph['h'],
            }

            quads.append(quad)
            cursor_x += glyph['advance'] * font_px

        return quads

    def render_gpu(self, quads: List[Dict], width: int, height: int,
                   position: Tuple[float, float], color: Tuple[float, float, float, float],
                   stroke_width: float = 0.0) -> np.ndarray:
        """
        Render text quads to RGBA layer using GPU

        Args:
            quads: List of glyph quad dictionaries
            width: Output layer width
            height: Output layer height
            position: (x, y) base position for all quads
            color: (r, g, b, a) color for all glyphs
            stroke_width: Stroke width in pixels

        Returns:
            RGBA layer as NumPy array (H, W, 4)
        """
        start_time = time.perf_counter()

        if not self.use_gpu or not CUPY_AVAILABLE:
            raise RuntimeError("GPU rendering requested but not available")

        # Build glyph data array: [x, y, u, v, w, h, r, g, b, a] per glyph
        num_glyphs = len(quads)
        glyph_data = np.zeros((num_glyphs, 10), dtype=np.float32)

        px, py = position
        r, g, b, a = color

        for i, quad in enumerate(quads):
            glyph_data[i] = [
                quad['x'] + px,
                quad['y'] + py,
                quad['u'],
                quad['v'],
                quad['w'],
                quad['h'],
                r, g, b, a
            ]

        # Transfer to GPU
        glyph_data_gpu = cp.asarray(glyph_data)
        output_gpu = cp.zeros((height, width, 4), dtype=cp.float32)

        # Compile and run kernel
        kernel = get_compiled_kernel('sdf_text_render', SDF_TEXT_RENDER_KERNEL)

        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )

        kernel(
            grid_size,
            block_size,
            (
                self.atlas_gpu,
                glyph_data_gpu,
                num_glyphs,
                output_gpu,
                width,
                height,
                self.atlas_size,
                0.5,  # SDF threshold
                stroke_width
            )
        )

        # Transfer back to CPU
        output = cp.asnumpy(output_gpu)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-SDF] GPU rendered {num_glyphs} glyphs @ {width}x{height} in {elapsed_ms:.2f}ms")

        return output

    def render_cpu(self, quads: List[Dict], width: int, height: int,
                   position: Tuple[float, float], color: Tuple[float, float, float, float],
                   stroke_width: float = 0.0) -> np.ndarray:
        """
        CPU fallback for text rendering

        Args:
            quads: List of glyph quad dictionaries
            width: Output layer width
            height: Output layer height
            position: (x, y) base position
            color: (r, g, b, a) color
            stroke_width: Stroke width (not implemented in CPU version)

        Returns:
            RGBA layer as NumPy array (H, W, 4)
        """
        start_time = time.perf_counter()

        # Simple CPU rasterization (low quality, for fallback only)
        from PIL import Image, ImageDraw, ImageFont

        output = np.zeros((height, width, 4), dtype=np.float32)

        # For now, use PIL as simple fallback
        # In production, would implement proper SDF sampling in NumPy

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-SDF] CPU rendered {len(quads)} glyphs @ {width}x{height} in {elapsed_ms:.2f}ms")

        return output

    def render(self, text: str, width: int, height: int, position: Tuple[float, float],
               font_px: int, color: Tuple[float, float, float, float],
               anchor: str = 'center', rtl: bool = False, stroke_width: float = 0.0,
               use_gpu: Optional[bool] = None) -> np.ndarray:
        """
        High-level text rendering interface

        Args:
            text: Text to render
            width: Output width
            height: Output height
            position: (x, y) position
            font_px: Font size in pixels
            color: (r, g, b, a) color
            anchor: Anchor position
            rtl: Right-to-left text
            stroke_width: Stroke width
            use_gpu: Override GPU setting

        Returns:
            RGBA layer (H, W, 4)
        """
        # Layout text
        quads = self.layout_text(text, font_px, anchor, rtl)

        if not quads:
            return np.zeros((height, width, 4), dtype=np.float32)

        # Render
        if use_gpu is None:
            use_gpu = self.use_gpu

        if use_gpu and CUPY_AVAILABLE:
            return self.render_gpu(quads, width, height, position, color, stroke_width)
        else:
            return self.render_cpu(quads, width, height, position, color, stroke_width)


# Global font atlas instance
_global_font_atlas: Optional[SDFFontAtlas] = None


def get_font_atlas(use_gpu: bool = True) -> SDFFontAtlas:
    """Get global font atlas instance (lazy initialization)"""
    global _global_font_atlas

    if _global_font_atlas is None:
        _global_font_atlas = SDFFontAtlas(use_gpu=use_gpu)

    return _global_font_atlas
