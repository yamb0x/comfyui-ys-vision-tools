"""
Layer Merge Node for YS-vision-tools
Merges multiple RGBA layers with various blend modes
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..utils import (
    ensure_numpy_hwc,
    alpha_blend,
    numpy_to_comfyui
)


class LayerMergeNode:
    """Merge multiple RGBA layers with blend modes"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "layer_1": ("LAYER",),
                "layer_2": ("LAYER",),
                "blend_mode": ([
                    "normal",
                    "add",
                    "multiply",
                    "screen",
                    "overlay",
                    "max",
                    "min",
                    "difference",
                ],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "layer_3": ("LAYER",),
                "layer_4": ("LAYER",),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Compositing"

    def execute(self, layer_1, layer_2, blend_mode, opacity, **kwargs):
        """Merge layers with specified blend mode"""

        # DEBUG
        print(f"\n[YS-MERGE] Executing LayerMerge")
        print(f"[YS-MERGE] layer_1 type: {type(layer_1)}, shape: {layer_1.shape if hasattr(layer_1, 'shape') else 'N/A'}")
        print(f"[YS-MERGE] layer_2 type: {type(layer_2)}, shape: {layer_2.shape if hasattr(layer_2, 'shape') else 'N/A'}")
        
        # Check if batch mode - detect tensors with batch dimension
        import torch
        is_batch = False
        batch_size = 1
        
        if torch.is_tensor(layer_1) and len(layer_1.shape) == 4 and layer_1.shape[0] > 1:
            is_batch = True
            batch_size = layer_1.shape[0]
        elif torch.is_tensor(layer_2) and len(layer_2.shape) == 4 and layer_2.shape[0] > 1:
            is_batch = True
            batch_size = layer_2.shape[0]
            
        if is_batch:
            print(f"[YS-MERGE] BATCH MODE: {batch_size} frames")
            batch_results = []
            
            for i in range(batch_size):
                # Get frame layers
                frame_l1 = layer_1[i:i+1] if torch.is_tensor(layer_1) and layer_1.shape[0] > 1 else layer_1
                frame_l2 = layer_2[i:i+1] if torch.is_tensor(layer_2) and layer_2.shape[0] > 1 else layer_2
                
                # Build frame kwargs
                frame_kwargs = {}
                if 'layer_3' in kwargs and kwargs['layer_3'] is not None:
                    l3 = kwargs['layer_3']
                    frame_kwargs['layer_3'] = l3[i:i+1] if torch.is_tensor(l3) and l3.shape[0] > 1 else l3
                if 'layer_4' in kwargs and kwargs['layer_4'] is not None:
                    l4 = kwargs['layer_4']
                    frame_kwargs['layer_4'] = l4[i:i+1] if torch.is_tensor(l4) and l4.shape[0] > 1 else l4
                
                # Process single frame
                result = self._merge_single_frame(frame_l1, frame_l2, blend_mode, opacity, **frame_kwargs)
                batch_results.append(result)
                print(f"[YS-MERGE] Frame {i}: merged")
            
            # Stack into batch
            batch_result = np.stack(batch_results, axis=0)
            print(f"[YS-MERGE] Returning batch: {batch_result.shape}")
            return (torch.from_numpy(batch_result.astype(np.float32)),)
        
        # Single frame mode
        print(f"[YS-MERGE] SINGLE MODE")
        result = self._merge_single_frame(layer_1, layer_2, blend_mode, opacity, **kwargs)
        return (numpy_to_comfyui(result),)

    def _merge_single_frame(self, layer_1, layer_2, blend_mode, opacity, **kwargs):
        """Merge single frame - extracted to avoid duplication"""

        # Convert to numpy arrays
        l1 = ensure_numpy_hwc(layer_1)
        l2 = ensure_numpy_hwc(layer_2)

        # Ensure RGBA
        l1 = self._ensure_rgba(l1)
        l2 = self._ensure_rgba(l2)

        # Resize layer_2 to match layer_1 if dimensions differ
        if l2.shape[:2] != l1.shape[:2]:
            import cv2
            target_height, target_width = l1.shape[:2]
            l2 = cv2.resize(l2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Apply opacity to layer 2
        l2 = l2.copy()
        l2[:, :, 3] *= opacity

        # Blend layers
        if blend_mode == "normal":
            result = alpha_blend(l2, l1)
        elif blend_mode == "add":
            result = self._blend_add(l1, l2)
        elif blend_mode == "multiply":
            result = self._blend_multiply(l1, l2)
        elif blend_mode == "screen":
            result = self._blend_screen(l1, l2)
        elif blend_mode == "overlay":
            result = self._blend_overlay(l1, l2)
        elif blend_mode == "difference":
            result = self._blend_difference(l1, l2)
        elif blend_mode == "max":
            result = self._blend_max(l1, l2)
        elif blend_mode == "min":
            result = self._blend_min(l1, l2)
        else:
            result = alpha_blend(l2, l1)

        # Merge additional layers if provided
        if 'layer_3' in kwargs and kwargs['layer_3'] is not None:
            l3 = ensure_numpy_hwc(kwargs['layer_3'])
            l3 = self._ensure_rgba(l3)

            # Resize to match result dimensions
            if l3.shape[:2] != result.shape[:2]:
                import cv2
                target_height, target_width = result.shape[:2]
                l3 = cv2.resize(l3, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            l3 = l3.copy()
            l3[:, :, 3] *= opacity
            result = alpha_blend(l3, result)

        if 'layer_4' in kwargs and kwargs['layer_4'] is not None:
            l4 = ensure_numpy_hwc(kwargs['layer_4'])
            l4 = self._ensure_rgba(l4)

            # Resize to match result dimensions
            if l4.shape[:2] != result.shape[:2]:
                import cv2
                target_height, target_width = result.shape[:2]
                l4 = cv2.resize(l4, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            l4 = l4.copy()
            l4[:, :, 3] *= opacity
            result = alpha_blend(l4, result)

        return result

    def _ensure_rgba(self, layer: np.ndarray) -> np.ndarray:
        """Ensure layer has RGBA channels"""
        if layer.shape[-1] == 3:
            # Add alpha channel
            alpha = np.ones((*layer.shape[:2], 1), dtype=layer.dtype)
            return np.concatenate([layer, alpha], axis=-1)
        return layer

    def _blend_add(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Additive blending"""
        result = base.copy()
        alpha = top[:, :, 3:4]

        result[:, :, :3] = np.clip(base[:, :, :3] + top[:, :, :3] * alpha, 0, 1)
        result[:, :, 3:4] = np.clip(base[:, :, 3:4] + top[:, :, 3:4], 0, 1)

        return result

    def _blend_multiply(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Multiply blending"""
        result = base.copy()
        alpha = top[:, :, 3:4]

        blended = base[:, :, :3] * top[:, :, :3]
        result[:, :, :3] = base[:, :, :3] * (1 - alpha) + blended * alpha
        result[:, :, 3:4] = np.clip(base[:, :, 3:4] + top[:, :, 3:4] * (1 - base[:, :, 3:4]), 0, 1)

        return result

    def _blend_screen(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Screen blending"""
        result = base.copy()
        alpha = top[:, :, 3:4]

        blended = 1 - (1 - base[:, :, :3]) * (1 - top[:, :, :3])
        result[:, :, :3] = base[:, :, :3] * (1 - alpha) + blended * alpha
        result[:, :, 3:4] = np.clip(base[:, :, 3:4] + top[:, :, 3:4] * (1 - base[:, :, 3:4]), 0, 1)

        return result

    def _blend_overlay(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Overlay blending"""
        result = base.copy()
        alpha = top[:, :, 3:4]

        mask = base[:, :, :3] < 0.5
        blended = np.where(mask,
                          2 * base[:, :, :3] * top[:, :, :3],
                          1 - 2 * (1 - base[:, :, :3]) * (1 - top[:, :, :3]))

        result[:, :, :3] = base[:, :, :3] * (1 - alpha) + blended * alpha
        result[:, :, 3:4] = np.clip(base[:, :, 3:4] + top[:, :, 3:4] * (1 - base[:, :, 3:4]), 0, 1)

        return result

    def _blend_max(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Maximum blending"""
        return np.maximum(base, top)

    def _blend_min(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Minimum blending"""
        return np.minimum(base, top)
    
    def _blend_difference(self, base: np.ndarray, top: np.ndarray) -> np.ndarray:
        """Difference blending (absolute difference)"""
        return np.abs(base - top)


NODE_CLASS_MAPPINGS = {
    "YS_LayerMerge": LayerMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_LayerMerge": "Layer Merge 🔀"
}