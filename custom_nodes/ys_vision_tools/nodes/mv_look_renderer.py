"""
Machine Vision Look Renderer Node - Phase 2

Apply machine vision / CRT / surveillance camera aesthetic effects.
Creates retro technical/surveillance looks with scanlines, chromatic aberration, etc.

Author: Yambo Studio
Part of: YS-vision-tools Phase 2 (Extended Renderers)
"""

import numpy as np
import torch
import time
from scipy.ndimage import shift
from typing import Dict, Any, Tuple, Optional

from ..utils import (
    numpy_to_comfyui,
    create_rgba_layer
)

# Try importing GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


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
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (10-50× faster for 4K)"
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
                    "default": "1.0,1.0,1.0",
                    "tooltip": "RGB tint (1.0,1.0,1.0 = no tint, 0.0,1.0,0.8 = surveillance green)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "LAYER")
    RETURN_NAMES = ("image", "layer")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Effects"

    def execute(self, image: torch.Tensor, scanline_intensity: float,
                chroma_offset_px: float, vignette: float, noise: float,
                as_layer: bool, use_gpu: bool, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply machine vision look effects.
        
        Supports batch processing for animations.

        Returns:
            image: Modified image (original if as_layer=True)
            layer: Effect layer (None if as_layer=False)
        """
        
        # DEBUG
        print(f"\n[YS-MVLOOK] Executing MVLook")
        print(f"[YS-MVLOOK] Input image shape: {image.shape}")
        print(f"[YS-MVLOOK] as_layer: {as_layer}")

        # Convert to numpy - PRESERVE BATCH DIMENSION
        # Don't use comfyui_to_numpy as it collapses batches!
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        print(f"[YS-MVLOOK] Numpy shape: {image_np.shape}")

        # Check if batch mode
        is_batch = len(image_np.shape) == 4
        
        if is_batch:
            print(f"[YS-MVLOOK] BATCH MODE: {image_np.shape[0]} frames")
            # Process each frame
            batch_size = image_np.shape[0]
            processed_frames = []
            
            for i in range(batch_size):
                frame = image_np[i]
                processed_frame = self._process_single_frame(
                    frame, scanline_intensity, chroma_offset_px, 
                    vignette, noise, use_gpu, **kwargs
                )
                processed_frames.append(processed_frame)
            
            result = np.stack(processed_frames, axis=0)
            print(f"[YS-MVLOOK] Processed batch result shape: {result.shape}")
        else:
            print(f"[YS-MVLOOK] SINGLE MODE")
            # Single frame
            result = self._process_single_frame(
                image_np, scanline_intensity, chroma_offset_px,
                vignette, noise, use_gpu, **kwargs
            )
            print(f"[YS-MVLOOK] Processed single result shape: {result.shape}")

        if as_layer:
            # Return as RGBA layer
            opacity = kwargs.get('opacity', 1.0)
            
            if is_batch:
                # Create batch of layers
                batch_size, h, w = result.shape[:3]
                layers = []
                
                for i in range(batch_size):
                    layer = create_rgba_layer(h, w)
                    layer[:, :, :3] = result[i]
                    layer[:, :, 3] = opacity
                    layers.append(layer)
                
                layer_batch = np.stack(layers, axis=0)
                print(f"[YS-MVLOOK] Returning batch layer: {layer_batch.shape}")
                # Convert to tensor preserving batch
                layer_tensor = torch.from_numpy(layer_batch.astype(np.float32))
                print(f"[YS-MVLOOK] Layer tensor shape: {layer_tensor.shape}")
                return (image, layer_tensor)
            else:
                # Single layer
                h, w = result.shape[:2]
                layer = create_rgba_layer(h, w)
                layer[:, :, :3] = result
                layer[:, :, 3] = opacity
                print(f"[YS-MVLOOK] Returning single layer: {layer.shape}")
                return (image, numpy_to_comfyui(layer))
        else:
            # Return modified image (both outputs always present)
            print(f"[YS-MVLOOK] Returning modified image: {result.shape}")
            # Convert to tensor preserving batch - DON'T use numpy_to_comfyui!
            result_tensor = torch.from_numpy(result.astype(np.float32))
            print(f"[YS-MVLOOK] Result tensor shape: {result_tensor.shape}")
            
            # Return modified image and empty layer (always return both outputs)
            if is_batch:
                batch_size, h, w = result.shape[:3]
                empty_layers = []
                for i in range(batch_size):
                    empty_layer = create_rgba_layer(h, w)
                    empty_layers.append(empty_layer)
                empty_batch = np.stack(empty_layers, axis=0)
                empty_tensor = torch.from_numpy(empty_batch.astype(np.float32))
                print(f"[YS-MVLOOK] Empty layer tensor shape: {empty_tensor.shape}")
                return (result_tensor, empty_tensor)
            else:
                h, w = result.shape[:2]
                empty_layer = create_rgba_layer(h, w)
                return (result_tensor, numpy_to_comfyui(empty_layer))
    
    def _process_single_frame(self, image: np.ndarray, scanline_intensity: float,
                             chroma_offset_px: float, vignette: float, 
                             noise: float, use_gpu: bool, **kwargs) -> np.ndarray:
        """Process a single frame with all effects - GPU or CPU path."""
        
        # GPU path
        if use_gpu and CUPY_AVAILABLE:
            start_time = time.perf_counter()
            result = self._process_frame_gpu(
                image, scanline_intensity, chroma_offset_px,
                vignette, noise, **kwargs
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[YS-MVLOOK] GPU processed frame in {elapsed_ms:.2f}ms")
            return result
        else:
            # Fallback warning
            if use_gpu and not CUPY_AVAILABLE:
                print("[YS-MVLOOK] GPU requested but CuPy unavailable, using CPU")
            
            start_time = time.perf_counter()
            result = self._process_frame_cpu(
                image, scanline_intensity, chroma_offset_px,
                vignette, noise, **kwargs
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[YS-MVLOOK] CPU processed frame in {elapsed_ms:.2f}ms")
            return result
    
    def _process_frame_gpu(self, image: np.ndarray, scanline_intensity: float,
                          chroma_offset_px: float, vignette: float,
                          noise: float, **kwargs) -> np.ndarray:
        """GPU-accelerated frame processing."""
        
        # Transfer to GPU
        img_gpu = cp.asarray(image, dtype=cp.float32)
        result_gpu = img_gpu
        
        # Apply effects on GPU
        if scanline_intensity > 0:
            result_gpu = self._apply_scanlines_gpu(
                result_gpu,
                scanline_intensity,
                kwargs.get('scanline_spacing', 2)
            )
        
        if chroma_offset_px > 0:
            result_gpu = self._apply_chromatic_aberration_gpu(result_gpu, chroma_offset_px)
        
        if vignette > 0:
            result_gpu = self._apply_vignette_gpu(result_gpu, vignette)
        
        if noise > 0:
            result_gpu = self._apply_noise_gpu(result_gpu, noise)
        
        # Apply color tint
        tint_str = kwargs.get('color_tint', '1.0,1.0,1.0')
        tint = self._parse_color(tint_str)
        if not np.allclose(tint, [1.0, 1.0, 1.0]):
            result_gpu = self._apply_color_tint_gpu(result_gpu, tint)
        
        # Transfer back to CPU
        result = cp.asnumpy(result_gpu)
        return result
    
    def _process_frame_cpu(self, image: np.ndarray, scanline_intensity: float,
                          chroma_offset_px: float, vignette: float,
                          noise: float, **kwargs) -> np.ndarray:
        """CPU fallback frame processing."""
        
        result = image.copy()

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
        tint_str = kwargs.get('color_tint', '1.0,1.0,1.0')
        tint = self._parse_color(tint_str)
        if not np.allclose(tint, [1.0, 1.0, 1.0]):
            result = self._apply_color_tint(result, tint)
        
        return result

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
    
    # ========================================
    # GPU-Accelerated Effect Methods
    # ========================================
    
    def _apply_scanlines_gpu(self, image_gpu, intensity: float, spacing: int):
        """GPU-accelerated scanline rendering."""
        h, w = image_gpu.shape[:2]
        
        # Create scanline mask on GPU
        scanline_mask = cp.ones((h, w), dtype=cp.float32)
        scanline_mask[::spacing, :] = 1.0 - intensity
        
        # Add phosphor glow on adjacent lines
        if spacing > 1:
            scanline_mask[1::spacing, :] = cp.minimum(1.0 + intensity * 0.2, 1.0)
        
        # Apply to all channels (GPU broadcast)
        result_gpu = image_gpu * scanline_mask[:, :, cp.newaxis]
        result_gpu = cp.clip(result_gpu, 0, 1)
        
        return result_gpu
    
    def _apply_chromatic_aberration_gpu(self, image_gpu, offset: float):
        """GPU-accelerated chromatic aberration using fast roll operation."""
        result_gpu = image_gpu.copy()
        
        if image_gpu.shape[2] >= 3:
            offset_int = int(offset)
            
            # Shift red channel left (roll negative)
            result_gpu[:, :, 0] = cp.roll(image_gpu[:, :, 0], -offset_int, axis=1)
            
            # Keep green centered (no change)
            # result_gpu[:, :, 1] = image_gpu[:, :, 1]
            
            # Shift blue channel right (roll positive)
            result_gpu[:, :, 2] = cp.roll(image_gpu[:, :, 2], offset_int, axis=1)
        
        return result_gpu
    
    def _apply_vignette_gpu(self, image_gpu, strength: float):
        """GPU-accelerated vignette with vectorized distance computation."""
        h, w = image_gpu.shape[:2]
        
        # Center point
        cy, cx = h / 2.0, w / 2.0
        
        # Create coordinate grids on GPU
        y = cp.arange(h, dtype=cp.float32)
        x = cp.arange(w, dtype=cp.float32)
        yy, xx = cp.meshgrid(y, x, indexing='ij')
        
        # Compute radial distance (normalized)
        max_dist = cp.sqrt(cx**2 + cy**2)
        dist = cp.sqrt((xx - cx)**2 + (yy - cy)**2) / max_dist
        
        # Quadratic falloff
        vignette_mask = 1.0 - (dist**2 * strength)
        vignette_mask = cp.clip(vignette_mask, 0, 1)
        
        # Apply to all channels (GPU broadcast)
        result_gpu = image_gpu * vignette_mask[:, :, cp.newaxis]
        
        return result_gpu
    
    def _apply_noise_gpu(self, image_gpu, amount: float):
        """GPU-accelerated noise generation."""
        # Generate Gaussian noise on GPU
        noise_gpu = cp.random.normal(0, amount, image_gpu.shape, dtype=cp.float32)
        
        # Add and clip
        result_gpu = image_gpu + noise_gpu
        result_gpu = cp.clip(result_gpu, 0, 1)
        
        return result_gpu
    
    def _apply_color_tint_gpu(self, image_gpu, tint: np.ndarray):
        """GPU-accelerated color tint."""
        # Convert tint to GPU array
        tint_gpu = cp.asarray(tint[:3], dtype=cp.float32)
        
        # Apply tint (GPU broadcast)
        result_gpu = image_gpu.copy()
        result_gpu[:, :, :3] *= tint_gpu
        result_gpu = cp.clip(result_gpu, 0, 1)
        
        return result_gpu

    def _parse_color(self, color_str: str) -> np.ndarray:
        """Parse color string to RGB array."""
        try:
            return np.array([float(x.strip()) for x in color_str.split(',')])[:3]
        except:
            return np.array([1.0, 1.0, 1.0])


# Node registration
NODE_CLASS_MAPPINGS = {"YS_MVLookRenderer": MVLookRendererNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_MVLookRenderer": "Machine Vision Look 📹"}