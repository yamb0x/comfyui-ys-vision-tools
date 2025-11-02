"""
Composite Over Node for YS-vision-tools
Composites RGBA layers over base images
"""

import numpy as np
from typing import Dict, Any

from ..utils import (
    ensure_numpy_hwc,
    alpha_blend,
    numpy_to_comfyui,
    resize_layer
)


class CompositeOverNode:
    """Composite RGBA layer over base image"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base_image": ("IMAGE",),
                "layer": ("LAYER",),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "x_offset": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "resize_to_base": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Compositing"

    def execute(self, base_image, layer, opacity, **kwargs):
        """Composite layer over base image"""

        # DEBUG
        print(f"\n[YS-COMPOSITE] Executing CompositeOver")
        print(f"[YS-COMPOSITE] base_image type: {type(base_image)}, shape: {base_image.shape if hasattr(base_image, 'shape') else 'N/A'}")
        print(f"[YS-COMPOSITE] layer type: {type(layer)}, shape: {layer.shape if hasattr(layer, 'shape') else 'N/A'}")
        
        # Check if batch mode
        import torch
        is_batch = False
        batch_size = 1
        
        if torch.is_tensor(base_image) and len(base_image.shape) == 4 and base_image.shape[0] > 1:
            is_batch = True
            batch_size = base_image.shape[0]
        elif torch.is_tensor(layer) and len(layer.shape) == 4 and layer.shape[0] > 1:
            is_batch = True
            batch_size = layer.shape[0]
            
        if is_batch:
            print(f"[YS-COMPOSITE] BATCH MODE: {batch_size} frames")
            batch_results = []
            
            for i in range(batch_size):
                # Get frame data
                frame_base = base_image[i:i+1] if torch.is_tensor(base_image) and base_image.shape[0] > 1 else base_image
                frame_layer = layer[i:i+1] if torch.is_tensor(layer) and layer.shape[0] > 1 else layer
                
                # Process single frame
                result = self._composite_single_frame(frame_base, frame_layer, opacity, **kwargs)
                batch_results.append(result)
                print(f"[YS-COMPOSITE] Frame {i}: composited")
            
            # Stack into batch
            batch_result = np.stack(batch_results, axis=0)
            print(f"[YS-COMPOSITE] Returning batch: {batch_result.shape}")
            return (torch.from_numpy(batch_result.astype(np.float32)),)
        
        # Single frame mode
        print(f"[YS-COMPOSITE] SINGLE MODE")
        result = self._composite_single_frame(base_image, layer, opacity, **kwargs)
        return (numpy_to_comfyui(result),)

    def _composite_single_frame(self, base_image, layer, opacity, **kwargs):
        """Composite single frame - extracted to avoid duplication"""

        # Convert to numpy arrays and ensure proper format
        base = ensure_numpy_hwc(base_image)
        overlay = ensure_numpy_hwc(layer)

        # Ensure float32 in [0, 1] range
        if base.dtype != np.float32:
            if base.max() > 1.0:
                base = base.astype(np.float32) / 255.0
            else:
                base = base.astype(np.float32)

        if overlay.dtype != np.float32:
            if overlay.max() > 1.0:
                overlay = overlay.astype(np.float32) / 255.0
            else:
                overlay = overlay.astype(np.float32)

        # Ensure base is RGB
        if len(base.shape) == 2:
            base = np.stack([base] * 3, axis=-1)
        elif base.shape[-1] == 4:
            base = base[:, :, :3]
        elif base.shape[-1] != 3:
            raise ValueError(f"Invalid base image shape: {base.shape}")

        # Ensure overlay is RGBA
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3 + [np.ones_like(overlay)], axis=-1)
        elif overlay.shape[-1] == 3:
            alpha = np.ones((*overlay.shape[:2], 1), dtype=overlay.dtype)
            overlay = np.concatenate([overlay, alpha], axis=-1)
        elif overlay.shape[-1] != 4:
            raise ValueError(f"Invalid overlay shape: {overlay.shape}")

        # Resize layer to match base if requested
        if kwargs.get('resize_to_base', True):
            if overlay.shape[:2] != base.shape[:2]:
                overlay = resize_layer(overlay, (base.shape[1], base.shape[0]), 'linear')

        # Apply global opacity
        overlay = overlay.copy()
        overlay[:, :, 3] *= opacity

        # Handle offsets
        x_offset = kwargs.get('x_offset', 0)
        y_offset = kwargs.get('y_offset', 0)

        if x_offset != 0 or y_offset != 0:
            overlay = self._apply_offset(overlay, x_offset, y_offset)

        # Convert base to RGBA for blending
        base_rgba = np.concatenate([
            base,
            np.ones((*base.shape[:2], 1), dtype=base.dtype)
        ], axis=-1)

        # Composite
        result = alpha_blend(overlay, base_rgba)

        # Remove alpha channel for output and ensure valid range
        result_rgb = np.clip(result[:, :, :3], 0.0, 1.0).astype(np.float32)

        # Return RGB for single frame
        return result_rgb

    def _apply_offset(self, layer: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
        """Apply position offset to layer"""

        h, w = layer.shape[:2]
        result = np.zeros_like(layer)

        # Calculate valid regions
        src_y_start = max(0, -y_offset)
        src_y_end = min(h, h - y_offset)
        src_x_start = max(0, -x_offset)
        src_x_end = min(w, w - x_offset)

        dst_y_start = max(0, y_offset)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, x_offset)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy valid region
        if src_y_end > src_y_start and src_x_end > src_x_start:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                layer[src_y_start:src_y_end, src_x_start:src_x_end]

        return result


NODE_CLASS_MAPPINGS = {
    "YS_CompositeOver": CompositeOverNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_CompositeOver": "Composite Over 🎬"
}