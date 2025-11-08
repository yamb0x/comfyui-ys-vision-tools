"""
Blur Region Renderer Node - Phase 2

Apply blur effects to regions around tracked points with smooth falloff.
Creates focused/defocused regions for artistic effects.

Author: Yambo Studio
Part of: YS-vision-tools Phase 2 (Extended Renderers)
"""

import numpy as np
import torch
import time
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional

from ..utils import create_rgba_layer

# Try importing GPU libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class BlurRegionRendererNode:
    """
    Apply blur effects to regions around tracked points.

    Features:
    - Gaussian blur with adjustable sigma
    - Smooth falloff masks (sigmoid)
    - Region radius control
    - Works with tracks or pre-computed boxes
    - Creates depth-of-field-like effects
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE tensor
                "radius_px": ("FLOAT", {
                    "default": 20.0,
                    "min": 5.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Blur region radius"
                }),
                "sigma_px": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Gaussian blur strength"
                }),
                "falloff": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Edge softness (0=hard, 1=very soft)"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blur layer opacity"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration for Gaussian blur (10-50× faster)"
                }),
            },
            "optional": {
                "tracks": ("TRACKS",),
                "boxes": ("BOXES",),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def execute(self, image: torch.Tensor, radius_px: float, sigma_px: float,
                falloff: float, opacity: float, use_gpu: bool, **kwargs):
        """
        Create blur layer with masked regions.

        Args:
            image: Input image tensor (BHWC format)
            radius_px: Blur region radius
            sigma_px: Gaussian blur sigma
            falloff: Edge softness
            opacity: Overall layer opacity

        Returns:
            LAYER: RGBA layer with blurred regions
        """

        # DEBUG
        print(f"\n[YS-BLUR] Executing BlurRegionRenderer")
        print(f"[YS-BLUR] Input image shape: {image.shape}")
        tracks = kwargs.get('tracks')
        print(f"[YS-BLUR] tracks type: {type(tracks)}")
        
        # Convert to numpy - PRESERVE BATCH DIMENSION
        # Don't use comfyui_to_numpy as it collapses batches!
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        print(f"[YS-BLUR] Numpy shape: {image_np.shape}")

        # Check if batch mode
        is_batch = len(image_np.shape) == 4
        batch_tracks = isinstance(tracks, list) if tracks is not None else False
        
        if is_batch:
            print(f"[YS-BLUR] BATCH MODE: {image_np.shape[0]} frames")
            batch_layers = []
            
            batch_size = image_np.shape[0]
            
            for i in range(batch_size):
                # Get frame image and tracks
                frame_img = image_np[i]
                frame_kwargs = kwargs.copy()
                if batch_tracks and i < len(tracks):
                    frame_kwargs['tracks'] = tracks[i]
                elif not batch_tracks and tracks is not None:
                    frame_kwargs['tracks'] = tracks
                
                # Process single frame
                layer = self._render_single_frame(
                    frame_img, radius_px, sigma_px, falloff, opacity, use_gpu, **frame_kwargs
                )
                batch_layers.append(layer)
                print(f"[YS-BLUR] Frame {i}: processed")
            
            # Stack into batch
            batch_result = np.stack(batch_layers, axis=0)
            print(f"[YS-BLUR] Returning batch: {batch_result.shape}")
            # Convert to tensor preserving batch
            return (torch.from_numpy(batch_result.astype(np.float32)),)

        # Single frame mode
        print(f"[YS-BLUR] SINGLE MODE")
        layer = self._render_single_frame(
            image_np, radius_px, sigma_px, falloff, opacity, use_gpu, **kwargs
        )
        return (torch.from_numpy(layer.astype(np.float32)).unsqueeze(0),)

    def _render_single_frame(self, image_np, radius_px, sigma_px, falloff, opacity, use_gpu, **kwargs):
        """Render single frame - extracted to avoid duplication"""

        h, w = image_np.shape[:2]

        # Create blur mask from track positions (GPU if available)
        mask = self._create_blur_mask(w, h, radius_px, falloff, use_gpu, **kwargs)

        # Apply gaussian blur to image
        # GPU path for significant speedup
        if use_gpu and CUPY_AVAILABLE:
            start_time = time.perf_counter()

            # Transfer to GPU
            img_gpu = cp.asarray(image_np)

            # GPU Gaussian blur (process all channels)
            blurred_gpu = cp.zeros_like(img_gpu)
            for c in range(img_gpu.shape[2]):
                blurred_gpu[:, :, c] = cupyx.scipy.ndimage.gaussian_filter(
                    img_gpu[:, :, c], sigma=sigma_px
                )

            # Transfer back to CPU
            blurred = cp.asnumpy(blurred_gpu)

            gpu_time = (time.perf_counter() - start_time) * 1000
            print(f"[YS-BLUR] GPU blurred image ({h}x{w}) in {gpu_time:.2f}ms")
        else:
            # CPU fallback
            if use_gpu and not CUPY_AVAILABLE:
                print("[YS-BLUR] GPU requested but CuPy not available, using CPU")

            start_time = time.perf_counter()
            blurred = np.zeros_like(image_np)
            for c in range(image_np.shape[2]):
                blurred[:, :, c] = gaussian_filter(image_np[:, :, c], sigma=sigma_px)
            cpu_time = (time.perf_counter() - start_time) * 1000
            print(f"[YS-BLUR] CPU blurred image ({h}x{w}) in {cpu_time:.2f}ms")

        # Create RGBA layer
        layer = create_rgba_layer(h, w)

        # Copy blurred regions to layer based on mask
        layer[:, :, :3] = blurred[:, :, :3]
        layer[:, :, 3] = mask * opacity

        return layer

    def _create_blur_mask(self, width: int, height: int,
                         radius: float, falloff: float, use_gpu: bool, **kwargs) -> np.ndarray:
        """
        Create mask for blur regions with smooth falloff.

        Args:
            width, height: Image dimensions
            radius: Blur region radius
            falloff: Edge softness factor
            use_gpu: Use GPU acceleration

        Returns:
            mask: 2D mask array (H, W) in range [0, 1]
        """

        # Get positions to blur
        positions = None

        if 'tracks' in kwargs and kwargs['tracks'] is not None:
            positions = kwargs['tracks']
            if isinstance(positions, list):
                positions = np.array(positions)

        elif 'boxes' in kwargs and kwargs['boxes'] is not None:
            # Use box centers
            boxes = kwargs['boxes']
            if isinstance(boxes, list):
                boxes = np.array(boxes)

            # boxes format: [x, y, w, h] or [x1, y1, x2, y2]
            if len(boxes) > 0:
                # Assume [x, y, w, h] format
                positions = boxes[:, :2] + boxes[:, 2:4] / 2

        if positions is None or len(positions) == 0:
            # Return empty mask
            return np.zeros((height, width), dtype=np.float32)

        # GPU path for mask creation (much faster for large images)
        if use_gpu and CUPY_AVAILABLE:
            start_time = time.perf_counter()

            # Create coordinate grids on GPU
            y_coords = cp.arange(height, dtype=cp.float32).reshape(-1, 1)
            x_coords = cp.arange(width, dtype=cp.float32).reshape(1, -1)

            # Initialize mask on GPU
            mask_gpu = cp.zeros((height, width), dtype=cp.float32)

            # Transfer positions to GPU
            positions_gpu = cp.asarray(positions, dtype=cp.float32)

            # Process all positions
            for i in range(len(positions_gpu)):
                x, y = positions_gpu[i]

                # Distance from point (vectorized on GPU)
                dist = cp.sqrt((x_coords - x)**2 + (y_coords - y)**2)

                # Apply smooth falloff using sigmoid function
                if falloff > 0:
                    scale = radius * falloff * 0.1 + 0.1
                    # Clip to prevent overflow in exp
                    exponent = cp.clip((dist - radius) / scale, -20, 20)
                    local_mask = 1.0 / (1.0 + cp.exp(exponent))
                else:
                    # Hard edge (no falloff)
                    local_mask = (dist <= radius).astype(cp.float32)

                # Combine with max (overlapping regions)
                mask_gpu = cp.maximum(mask_gpu, local_mask)

            # Transfer back to CPU
            mask = cp.asnumpy(mask_gpu)

            gpu_time = (time.perf_counter() - start_time) * 1000
            print(f"[YS-BLUR] GPU created mask ({len(positions)} points) in {gpu_time:.2f}ms")

            return mask

        # CPU fallback
        start_time = time.perf_counter()

        mask = np.zeros((height, width), dtype=np.float32)

        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:height, :width]

        for x, y in positions:
            # Distance from point
            dist = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)

            # Apply smooth falloff using sigmoid function
            if falloff > 0:
                # Sigmoid falloff for smooth transitions
                scale = radius * falloff * 0.1 + 0.1
                # Clip to prevent overflow in exp
                exponent = np.clip((dist - radius) / scale, -20, 20)
                local_mask = 1.0 / (1.0 + np.exp(exponent))
            else:
                # Hard edge (no falloff)
                local_mask = (dist <= radius).astype(float)

            # Combine with max (overlapping regions)
            mask = np.maximum(mask, local_mask)

        cpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-BLUR] CPU created mask ({len(positions)} points) in {cpu_time:.2f}ms")

        return mask


# Node registration
NODE_CLASS_MAPPINGS = {"YS_BlurRegionRenderer": BlurRegionRendererNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_BlurRegionRenderer": "Blur Region Renderer 🌫️"}