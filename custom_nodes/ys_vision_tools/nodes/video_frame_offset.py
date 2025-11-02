"""
Video Frame Offset Node - Phase 1.5

Provides frame offset capability for video workflows, enabling motion detection
and temporal effects. Essential for optical flow and motion-based tracking.

Author: Yambo Studio
Part of: YS-vision-tools Phase 1.5 (UX Polish & Video Support)
"""

import torch
import numpy as np
from typing import Tuple, Literal


class VideoFrameOffsetNode:
    """
    Extract offset frames from video batches for temporal operations.

    This node enables motion detection workflows by providing access to
    previous frames. It's essential for optical flow detection and creating
    motion-based visual effects.

    Features:
    - Frame offset from -10 to 0 (negative = previous frames)
    - Multiple edge case handling modes
    - Valid mask output for frame availability checking
    - Clean integration with Track Detect optical flow

    Workflow Pattern:
        Load Video → Video Frame Offset → Track Detect (optical_flow)
                         ↓ (current + previous frames)
                    Line Link Renderer (motion trails)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of frames from video
                "offset": ("INT", {
                    "default": -1,
                    "min": -10,
                    "max": 0,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Frame offset (-1 = previous frame, -2 = 2 frames back)"
                }),
                "mode": (["hold_first", "repeat_edge", "black_frame"], {
                    "default": "hold_first",
                    "tooltip": "How to handle first frame when no previous exists"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("current_frame", "offset_frame", "valid_mask")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Video"

    def execute(
        self,
        images: torch.Tensor,
        offset: int = -1,
        mode: Literal["hold_first", "repeat_edge", "black_frame"] = "hold_first"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract offset frames from video batch.

        Args:
            images: Video batch tensor (B, H, W, C) where B is number of frames
            offset: Frame offset (-1 = previous frame, -2 = 2 frames back, etc.)
            mode: How to handle edge cases when offset frame doesn't exist:
                - "hold_first": Use first frame as previous
                - "repeat_edge": Use current frame as previous
                - "black_frame": Use black frame as previous

        Returns:
            Tuple of:
                - current_frames: Current frame batch (B, H, W, C)
                - offset_frames: Previous frame batch (B, H, W, C)
                - valid_masks: Validity mask (B, 1, 1, 1) - True if offset frame is valid

        Examples:
            >>> # For optical flow with previous frame
            >>> current, previous, valid = node.execute(video_batch, offset=-1)
            >>>
            >>> # For 2-frame temporal difference
            >>> current, two_back, valid = node.execute(video_batch, offset=-2)

        Edge Cases:
            - Frame 0 with offset=-1: No previous frame exists
              - hold_first: previous = frame 0
              - repeat_edge: previous = frame 0
              - black_frame: previous = zeros
            - Valid mask indicates if offset was valid (False for first frame)
        """

        batch_size = images.shape[0]
        height, width, channels = images.shape[1:]

        # Preallocate output tensors
        current_frames = []
        offset_frames = []
        valid_masks = []

        for i in range(batch_size):
            # Current frame is always valid
            current_frame = images[i:i+1]  # Keep batch dimension
            current_frames.append(current_frame)

            # Calculate offset index
            offset_idx = i + offset

            # Handle edge cases where offset frame doesn't exist
            if offset_idx < 0:
                # Out of bounds - no valid previous frame
                if mode == "hold_first":
                    # Use first frame as "previous"
                    offset_frame = images[0:1]
                elif mode == "repeat_edge":
                    # Use current frame as "previous" (no motion)
                    offset_frame = current_frame
                else:  # black_frame
                    # Use black frame
                    offset_frame = torch.zeros_like(current_frame)

                valid = False
            else:
                # Valid offset frame exists
                offset_frame = images[offset_idx:offset_idx+1]
                valid = True

            offset_frames.append(offset_frame)

            # Create validity mask (shape: 1, 1, 1, 1 for broadcasting)
            valid_mask = torch.tensor([[[[float(valid)]]]], dtype=torch.float32)
            valid_masks.append(valid_mask)

        # Concatenate all frames
        current_batch = torch.cat(current_frames, dim=0)
        offset_batch = torch.cat(offset_frames, dim=0)
        valid_batch = torch.cat(valid_masks, dim=0)

        return (current_batch, offset_batch, valid_batch)


# Node registration will be handled in __init__.py
NODE_CLASS_MAPPINGS = {"YS_VideoFrameOffset": VideoFrameOffsetNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YS_VideoFrameOffset": "Video Frame Offset 🎬"}
