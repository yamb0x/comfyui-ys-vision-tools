"""
Image Size Detector Node - Phase 1.5

Automatically detects and outputs image/video dimensions for dynamic workflows.
Eliminates the need for manual size entry and prevents dimension mismatch errors.

Author: Yambo Studio
Part of: YS-vision-tools Phase 1.5 (UX Polish & Video Support)
"""

import torch
import numpy as np
from typing import Tuple


class ImageSizeDetectorNode:
    """
    Detect image dimensions and pass them through as separate outputs.

    This node is essential for creating dynamic workflows that adapt to any
    image size without manual configuration. It's especially useful for video
    workflows where dimensions may vary.

    Features:
    - Auto-detects width and height from image tensor
    - Provides both INT outputs (for node inputs) and STRING (for display)
    - Pass-through design (image flows through unchanged)
    - Works with both images and video batches

    Workflow Pattern:
        Load Image → Image Size Detector → [width, height outputs]
                           ↓ (image pass-through)
                      Track Detect → Line Link Renderer
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE type (BHWC format)
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "dimensions")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Utilities"

    def execute(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int, str]:
        """
        Extract dimensions from ComfyUI image tensor.

        Args:
            image: ComfyUI IMAGE tensor in BHWC format (Batch, Height, Width, Channels)

        Returns:
            Tuple of:
                - image: Pass-through of input image (unchanged)
                - width: Image width in pixels (INT)
                - height: Image height in pixels (INT)
                - dimensions: Formatted string "WxH" (e.g., "1920x1080")

        Notes:
            - ComfyUI uses BHWC format: (Batch, Height, Width, Channels)
            - Width is dimension 2, Height is dimension 1
            - For video batches, dimensions are extracted from first frame
        """

        # Extract shape from tensor
        # image.shape = (B, H, W, C)
        batch_size, height, width, channels = image.shape

        # Create formatted dimension string
        dimensions_str = f"{width}x{height}"

        # Return image pass-through + dimensional data
        return (
            image,           # Pass-through for next node
            width,           # INT for node inputs
            height,          # INT for node inputs
            dimensions_str   # STRING for display/debugging
        )


# Node registration will be handled in __init__.py
NODE_CLASS_MAPPINGS = {"YS_ImageSizeDetector": ImageSizeDetectorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_ImageSizeDetector": "Image Size Detector 📐"}
