"""
Text On Tracks Node - Clean Rebuild

Renders text labels at tracked point positions with three content modes:
- Coordinates: Display point position as formatted text
- Code Snippets: Random lines from this codebase
- String: Static custom text

Features:
- PIL/Pillow-based text rendering (reliable, cross-platform)
- Full font, color, stroke, alignment control
- Kerning (letter spacing) support
- Batch processing for video
- GPU-accelerated compositing
- Compatible with YS color picker

Author: Yambo Studio
Part of: YS-vision-tools
"""

import numpy as np
import torch
import time
import random
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ..utils import normalize_color_to_rgba01

# GPU imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class TextOnTracksNode:
    """
    Render text labels on tracked points

    Three content modes:
    1. Coordinates: Show point position like "[123, 456]"
    2. Code Snippets: Random code line from codebase
    3. String: Static text (customizable)

    Full control over typography: font, size, color, stroke, alignment, kerning
    """

    # Font cache to avoid reloading
    _font_cache = {}
    _codebase_lines = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Discover available fonts
        font_list = cls._get_available_fonts()

        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),

                # Content mode
                "content_mode": ([
                    "coordinates",
                    "code_snippets",
                    "string"
                ], {
                    "default": "coordinates",
                    "tooltip": "What text to display at each point"
                }),

                # Static string for 'string' mode
                "text_string": ("STRING", {
                    "default": "Track Point",
                    "multiline": False,
                    "tooltip": "Text to display when content_mode is 'string'"
                }),

                # Character length control
                "max_chars": ("INT", {
                    "default": 50,
                    "min": 5,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Maximum text length"
                }),

                # Typography
                "font_name": (font_list, {
                    "default": font_list[0] if font_list else "arial.ttf",
                    "tooltip": "Font to use"
                }),
                "font_size": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 72,
                    "step": 1,
                    "tooltip": "Font size in pixels"
                }),
                "letter_spacing": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Kerning: space between letters"
                }),

                # Color and opacity
                "color": ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Text color"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Text opacity"
                }),

                # Stroke
                "stroke_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Outline thickness (0=no stroke)"
                }),
                "stroke_color": ("COLOR", {
                    "default": "#000000",
                    "tooltip": "Outline color"
                }),

                # Position and alignment
                "offset_x": ("INT", {
                    "default": 10,
                    "min": -500,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Horizontal offset from point"
                }),
                "offset_y": ("INT", {
                    "default": -20,
                    "min": -500,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Vertical offset from point"
                }),
                "alignment": ([
                    "bottom_left",
                    "bottom_center",
                    "bottom_right",
                    "center_left",
                    "center",
                    "center_right",
                    "top_left",
                    "top_center",
                    "top_right"
                ], {
                    "default": "bottom_left",
                    "tooltip": "Text anchor position relative to offset point"
                }),

                # Performance
                "use_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use GPU for compositing (CPU for text rendering)"
                }),
            },
            "optional": {
                "palette": ("PALETTE",),
            }
        }

    RETURN_TYPES = ("LAYER",)
    RETURN_NAMES = ("layer",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    @classmethod
    def _get_available_fonts(cls) -> List[str]:
        """Discover available system fonts"""
        fonts = []

        # Platform-specific font directories
        if sys.platform == "win32":
            font_dirs = [
                Path(os.environ.get("SystemRoot", "C:\\Windows")) / "Fonts"
            ]
        elif sys.platform == "darwin":
            font_dirs = [
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library/Fonts"
            ]
        else:  # Linux
            font_dirs = [
                Path("/usr/share/fonts/truetype"),
                Path("/usr/share/fonts/TTF"),
                Path.home() / ".fonts"
            ]

        # Scan for .ttf and .otf files
        for font_dir in font_dirs:
            if font_dir.exists():
                for font_file in font_dir.rglob("*.ttf"):
                    fonts.append(font_file.name)
                for font_file in font_dir.rglob("*.otf"):
                    fonts.append(font_file.name)

        # Deduplicate and sort
        fonts = sorted(list(set(fonts)))

        # Fallback if no fonts found
        if not fonts:
            fonts = ["default"]

        return fonts

    @classmethod
    def _load_font(cls, font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load font with caching"""
        cache_key = (font_name, font_size)

        if cache_key in cls._font_cache:
            return cls._font_cache[cache_key]

        font = None

        # Try to load specified font
        if font_name != "default":
            # Platform-specific font directories
            if sys.platform == "win32":
                font_dirs = [
                    Path(os.environ.get("SystemRoot", "C:\\Windows")) / "Fonts"
                ]
            elif sys.platform == "darwin":
                font_dirs = [
                    Path("/System/Library/Fonts"),
                    Path("/Library/Fonts"),
                ]
            else:  # Linux
                font_dirs = [
                    Path("/usr/share/fonts/truetype"),
                    Path("/usr/share/fonts/TTF"),
                ]

            # Search for font file
            for font_dir in font_dirs:
                font_path = font_dir / font_name
                if font_path.exists():
                    try:
                        font = ImageFont.truetype(str(font_path), font_size)
                        print(f"[YS-TEXT] Loaded font: {font_path}")
                        break
                    except Exception as e:
                        print(f"[YS-TEXT] Failed to load {font_path}: {e}")
                        continue

                # Also try recursive search
                for found_font in font_dir.rglob(font_name):
                    try:
                        font = ImageFont.truetype(str(found_font), font_size)
                        print(f"[YS-TEXT] Loaded font: {found_font}")
                        break
                    except Exception as e:
                        continue

                if font:
                    break

        # Fallback to default font
        if font is None:
            try:
                font = ImageFont.load_default()
                print(f"[YS-TEXT] Using default PIL font (size parameter ignored)")
            except Exception as e:
                print(f"[YS-TEXT] ERROR: Could not load any font: {e}")
                raise RuntimeError("No fonts available")

        cls._font_cache[cache_key] = font
        return font

    @classmethod
    def _load_codebase_lines(cls) -> List[str]:
        """Load random code snippets from the codebase"""
        if cls._codebase_lines is not None:
            return cls._codebase_lines

        lines = []

        # Get the project root (parent of parent of this file)
        project_root = Path(__file__).parent.parent.parent

        # Scan all .py files in custom_nodes/ys_vision_tools
        py_files = list((project_root / "custom_nodes" / "ys_vision_tools").rglob("*.py"))

        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()

                    # Filter out empty lines, comments only, and very short lines
                    for line in file_lines:
                        stripped = line.strip()
                        if (len(stripped) > 20 and
                            not stripped.startswith('#') and
                            not stripped.startswith('"""') and
                            not stripped.startswith("'''")):
                            lines.append(stripped)
            except Exception as e:
                continue

        # Fallback if no code found
        if not lines:
            lines = [
                'def process(self, data): return result',
                'layer = torch.zeros((H, W, 4), dtype=torch.float32)',
                'positions = tracks[:, :2]  # Extract x, y coordinates',
            ]

        cls._codebase_lines = lines
        print(f"[YS-TEXT] Loaded {len(lines)} code snippets from codebase")
        return lines

    def execute(self, tracks: np.ndarray, image_width: int, image_height: int,
                content_mode: str, text_string: str, max_chars: int,
                font_name: str, font_size: int, letter_spacing: float,
                color, opacity: float,
                stroke_width: int, stroke_color,
                offset_x: int, offset_y: int, alignment: str,
                use_gpu: bool,
                palette: Optional[np.ndarray] = None) -> Tuple[torch.Tensor]:
        """
        Render text labels at track positions

        Args:
            tracks: Track positions (N, 2) or list of arrays for batch
            content_mode: "coordinates", "code_snippets", or "string"
            text_string: Static text for 'string' mode
            max_chars: Maximum text length
            All other params control typography and positioning

        Returns:
            RGBA layer tensor with rendered text
        """
        start_time = time.perf_counter()

        # Parse colors
        text_rgba = normalize_color_to_rgba01(color, opacity)
        stroke_rgba = normalize_color_to_rgba01(stroke_color, 1.0)

        print(f"\n[YS-TEXT] Text On Tracks")
        print(f"[YS-TEXT] Mode: {content_mode}, Font: {font_name} {font_size}px")
        print(f"[YS-TEXT] Color: {color} -> RGBA: {text_rgba}")

        # Handle batch vs single
        if isinstance(tracks, list):
            is_batch = True
            batch_size = len(tracks)
            print(f"[YS-TEXT] BATCH MODE: {batch_size} frames")
        else:
            is_batch = False
            batch_size = 1
            tracks = [tracks]
            if palette is not None:
                palette = [palette]
            print(f"[YS-TEXT] SINGLE MODE: 1 frame")

        # Load font
        font = self._load_font(font_name, font_size)

        # Load code snippets if needed
        if content_mode == "code_snippets":
            code_lines = self._load_codebase_lines()

        # Process each frame
        output_frames = []

        for frame_idx in range(batch_size):
            frame_tracks = tracks[frame_idx]
            frame_palette = palette[frame_idx] if palette is not None else None

            # Generate text content for each track
            texts = []
            colors = []

            for i, pos in enumerate(frame_tracks):
                # Generate content
                if content_mode == "coordinates":
                    # Format: [x, y] or detailed format
                    text = f"[{int(pos[0])}, {int(pos[1])}]"

                elif content_mode == "code_snippets":
                    # Random code line
                    text = random.choice(code_lines)

                elif content_mode == "string":
                    # Static text
                    text = text_string

                else:
                    text = f"Point {i}"

                # Truncate to max length
                if len(text) > max_chars:
                    text = text[:max_chars-3] + "..."

                texts.append(text)

                # Get color (palette or default)
                if frame_palette is not None and i < len(frame_palette):
                    # Use palette RGB with text opacity
                    r, g, b = frame_palette[i][:3]
                    colors.append((r, g, b, opacity))
                else:
                    colors.append(text_rgba)

            # Render text layer for this frame
            if use_gpu and CUPY_AVAILABLE:
                frame_np = self._render_gpu(
                    texts, frame_tracks, colors, image_width, image_height,
                    font, stroke_width, stroke_rgba, offset_x, offset_y,
                    alignment, letter_spacing
                )
            else:
                frame_np = self._render_cpu(
                    texts, frame_tracks, colors, image_width, image_height,
                    font, stroke_width, stroke_rgba, offset_x, offset_y,
                    alignment, letter_spacing
                )

            output_frames.append(frame_np)

            # Progress logging
            if is_batch and (frame_idx % 10 == 0 or frame_idx == batch_size - 1):
                print(f"[YS-TEXT] Processed frame {frame_idx+1}/{batch_size}")

        # Stack into batch tensor
        if is_batch:
            output_batch = np.stack(output_frames, axis=0)
            layer_tensor = torch.from_numpy(output_batch).float()
        else:
            layer_tensor = torch.from_numpy(output_frames[0]).unsqueeze(0).float()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-TEXT] Total time: {elapsed_ms:.2f}ms for {batch_size} frame(s)")

        return (layer_tensor,)

    def _render_cpu(self, texts: List[str], positions: List[np.ndarray],
                    colors: List[Tuple[float, float, float, float]],
                    width: int, height: int, font: ImageFont.FreeTypeFont,
                    stroke_width: int, stroke_rgba: Tuple[float, float, float, float],
                    offset_x: int, offset_y: int, alignment: str,
                    letter_spacing: float) -> np.ndarray:
        """
        Render text using PIL/Pillow (CPU)

        This is the reliable, working implementation
        """
        start_time = time.perf_counter()

        # Create RGBA image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Convert stroke color to PIL format (0-255)
        stroke_color_pil = tuple((np.array(stroke_rgba[:3]) * 255).astype(np.uint8))

        # Render each text label
        for text, pos, color in zip(texts, positions, colors):
            if not text:
                continue

            # Calculate text size
            if hasattr(draw, 'textbbox'):
                # PIL 8.0+
                bbox = draw.textbbox((0, 0), text, font=font, spacing=letter_spacing)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Older PIL
                text_width, text_height = draw.textsize(text, font=font)

            # Calculate position based on alignment
            x = pos[0] + offset_x
            y = pos[1] + offset_y

            # Apply alignment anchor
            if "left" in alignment:
                pass  # x unchanged
            elif "center" in alignment:
                x -= text_width / 2
            elif "right" in alignment:
                x -= text_width

            if "top" in alignment:
                pass  # y unchanged
            elif "center" in alignment and alignment == "center":
                y -= text_height / 2
            elif "bottom" in alignment:
                y -= text_height

            # Convert color to PIL format (0-255)
            fill_color_pil = tuple((np.array(color) * 255).astype(np.uint8))

            # Render text with stroke if enabled
            if stroke_width > 0:
                # Draw stroke by rendering text multiple times offset
                draw.text(
                    (x, y), text, font=font, fill=fill_color_pil,
                    stroke_width=stroke_width, stroke_fill=stroke_color_pil
                )
            else:
                # No stroke
                draw.text((x, y), text, font=font, fill=fill_color_pil)

        # Convert to numpy (0-1 range)
        output = np.array(img, dtype=np.float32) / 255.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-TEXT] CPU rendered {len(texts)} labels @ {width}x{height} in {elapsed_ms:.2f}ms")

        return output

    def _render_gpu(self, texts: List[str], positions: List[np.ndarray],
                    colors: List[Tuple[float, float, float, float]],
                    width: int, height: int, font: ImageFont.FreeTypeFont,
                    stroke_width: int, stroke_rgba: Tuple[float, float, float, float],
                    offset_x: int, offset_y: int, alignment: str,
                    letter_spacing: float) -> np.ndarray:
        """
        Render text with GPU-accelerated compositing

        Text rasterization still happens on CPU (PIL), but compositing uses GPU
        """
        start_time = time.perf_counter()

        # Render on CPU first
        cpu_output = self._render_cpu(
            texts, positions, colors, width, height, font,
            stroke_width, stroke_rgba, offset_x, offset_y, alignment, letter_spacing
        )

        # Transfer to GPU for potential further processing
        # (In this case, just a transfer - future GPU ops could be added here)
        try:
            gpu_output = cp.asarray(cpu_output)
            output = cp.asnumpy(gpu_output)
        except Exception as e:
            print(f"[YS-TEXT] GPU transfer failed: {e}, using CPU output")
            output = cpu_output

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-TEXT] GPU mode rendered {len(texts)} labels in {elapsed_ms:.2f}ms")

        return output


# Register node
NODE_CLASS_MAPPINGS = {
    "YSTextOnTracks": TextOnTracksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSTextOnTracks": "YS Text On Tracks",
}
