"""
Blur Region Renderer Node - Phase 2

Apply blur effects to regions around tracked points with smooth falloff.
Creates focused/defocused regions for artistic effects.

Author: Yambo Studio
Part of: YS-vision-tools Phase 2 (Extended Renderers)
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional

from ..utils import (
    create_rgba_layer,
    comfyui_to_numpy,
    numpy_to_comfyui
)


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
                falloff: float, opacity: float, **kwargs):
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
        tracks = kwargs.get('tracks')
        print(f"[YS-BLUR] tracks type: {type(tracks)}")
        
        # Convert ComfyUI tensor to numpy
        image_np = comfyui_to_numpy(image)
        print(f"[YS-BLUR] image_np shape: {image_np.shape}")

        # Check if batch mode
        is_batch = len(image_np.shape) == 4 and image_np.shape[0] > 1
        batch_tracks = isinstance(tracks, list) if tracks is not None else False
        
        if is_batch or batch_tracks:
            print(f"[YS-BLUR] BATCH MODE: {image_np.shape[0] if is_batch else len(tracks)} frames")
            batch_layers = []
            
            num_frames = image_np.shape[0] if is_batch else (len(tracks) if batch_tracks else 1)
            
            for i in range(num_frames):
                # Get frame image and tracks
                frame_img = image_np[i] if is_batch else image_np[0]
                frame_kwargs = kwargs.copy()
                if batch_tracks:
                    frame_kwargs['tracks'] = tracks[i]
                
                # Process single frame
                layer = self._render_single_frame(
                    frame_img, radius_px, sigma_px, falloff, opacity, **frame_kwargs
                )
                batch_layers.append(layer)
                print(f"[YS-BLUR] Frame {i}: processed")
            
            # Stack into batch
            batch_result = np.stack(batch_layers, axis=0)
            print(f"[YS-BLUR] Returning batch: {batch_result.shape}")
            # Don't use numpy_to_comfyui - already in BHWC format, just convert to tensor
            import torch
            return (torch.from_numpy(batch_result.astype(np.float32)),)

        # Single frame mode
        print(f"[YS-BLUR] SINGLE MODE")
        if len(image_np.shape) == 4:
            image_np = image_np[0]
        
        layer = self._render_single_frame(
            image_np, radius_px, sigma_px, falloff, opacity, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, image_np, radius_px, sigma_px, falloff, opacity, **kwargs):
        """Render single frame - extracted to avoid duplication"""

        h, w = image_np.shape[:2]

        # Create blur mask from track positions
        mask = self._create_blur_mask(w, h, radius_px, falloff, **kwargs)

        # Apply gaussian blur to image
        # Apply per-channel to handle RGB
        blurred = np.zeros_like(image_np)
        for c in range(image_np.shape[2]):
            blurred[:, :, c] = gaussian_filter(image_np[:, :, c], sigma=sigma_px)

        # Create RGBA layer
        layer = create_rgba_layer(h, w)

        # Copy blurred regions to layer based on mask
        layer[:, :, :3] = blurred[:, :, :3]
        layer[:, :, 3] = mask * opacity

        return layer

    def _create_blur_mask(self, width: int, height: int,
                         radius: float, falloff: float, **kwargs) -> np.ndarray:
        """
        Create mask for blur regions with smooth falloff.

        Args:
            width, height: Image dimensions
            radius: Blur region radius
            falloff: Edge softness factor

        Returns:
            mask: 2D mask array (H, W) in range [0, 1]
        """

        mask = np.zeros((height, width), dtype=np.float32)

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
            return mask

        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:height, :width]

        for x, y in positions:
            # Distance from point
            dist = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)

            # Apply smooth falloff using sigmoid function
            if falloff > 0:
                # Sigmoid falloff for smooth transitions
                # Scale controls steepness
                scale = radius * falloff * 0.1 + 0.1  # Add small constant to avoid division by zero
                local_mask = 1.0 / (1.0 + np.exp((dist - radius) / scale))
            else:
                # Hard edge (no falloff)
                local_mask = (dist <= radius).astype(float)

            # Combine with max (overlapping regions)
            mask = np.maximum(mask, local_mask)

        return mask


# Node registration
NODE_CLASS_MAPPINGS = {"YS_BlurRegionRenderer": BlurRegionRendererNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_BlurRegionRenderer": "Blur Region Renderer 🌫️"}