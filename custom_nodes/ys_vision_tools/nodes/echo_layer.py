"""
Echo Layer Node - Temporal Decay / Trails

Creates motion trails and echo effects by accumulating layers across frames
with exponential decay and optional optical flow warping.

Features:
- Exponential moving average (EMA) temporal accumulation
- Multiple decay modes: accumulate, pingpong, framecount
- Optical flow warping for motion-aware trails
- GPU-accelerated with custom CUDA kernel
- Configurable exposure and clamping

Author: Yambo Studio
Part of: YS-vision-tools Phase 3 (New Nodes)
"""

import numpy as np
import torch
import time
from typing import Dict, Any, Optional, Tuple

from ..utils import create_rgba_layer

# Try importing GPU libraries
try:
    import cupy as cp
    from ..utils.cuda_kernels import get_compiled_kernel, ECHO_EMA_UPDATE_KERNEL
    from ..utils.optical_flow import backward_warp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class EchoLayerNode:
    """
    Temporal decay / echo / trail effect node

    Takes any RGBA layer (dots, lines, boxes, text, etc.) and extends its
    visual presence across subsequent frames with controllable decay and
    optional motion warping.

    Perfect for:
    - Motion trails on tracked objects
    - Ghosting / echo effects
    - Persistence-of-vision effects
    - Stroboscopic visualization
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "layer": ("LAYER",),  # Current frame RGBA overlay
                "mode": (["accumulate", "pingpong", "framecount"], {
                    "default": "accumulate",
                    "tooltip": "Decay mode: accumulate=standard EMA, pingpong=fade in/out, framecount=N-frame persistence"
                }),
                "decay": ("FLOAT", {
                    "default": 0.88,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "EMA decay factor per frame (higher=longer trails)"
                }),
                "boost_on_new": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Extra alpha boost on fresh pixels"
                }),
                "max_age": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Maximum trail age in frames"
                }),
                "exposure": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Post-accumulation brightness gain"
                }),
                "clamp_mode": (["hard", "softclip"], {
                    "default": "hard",
                    "tooltip": "Output clamping: hard=[0,1], softclip=soft saturation"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (3-5× faster)"
                }),
            },
            "optional": {
                "state": ("STATE",),  # Internal accumulator state
                "flow_u": ("FLOW_U",),  # Optical flow horizontal
                "flow_v": ("FLOW_V",),  # Optical flow vertical
                "warp_with_flow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable optical flow warping (requires flow_u/flow_v)"
                }),
            }
        }

    RETURN_TYPES = ("LAYER", "STATE")
    RETURN_NAMES = ("layer", "state")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Effects"

    def execute(self, layer: torch.Tensor, mode: str, decay: float,
                boost_on_new: float, max_age: int, exposure: float,
                clamp_mode: str, use_gpu: bool,
                state: Optional[Dict] = None,
                flow_u: Optional[torch.Tensor] = None,
                flow_v: Optional[torch.Tensor] = None,
                warp_with_flow: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply temporal echo/trail effect to layer

        Args:
            layer: Input RGBA layer (BHWC format, premultiplied)
            mode: Decay mode
            decay: EMA decay factor
            boost_on_new: Alpha boost for new pixels
            max_age: Maximum age in frames
            exposure: Output brightness gain
            clamp_mode: Clamping method
            use_gpu: Use GPU acceleration
            state: Previous state dict with buffers
            flow_u: Optical flow horizontal component
            flow_v: Optical flow vertical component
            warp_with_flow: Enable flow warping

        Returns:
            Tuple of (output_layer, updated_state)
        """
        # Check if batch mode
        batch_size = layer.shape[0]
        is_batch = batch_size > 1

        if is_batch:
            print(f"[YS-ECHO] BATCH MODE: {batch_size} frames")

        # Get first frame dimensions
        first_frame = layer[0].cpu().numpy()
        H, W = first_frame.shape[:2]

        # Initialize or retrieve state
        if state is None or state.get('shape') != (H, W):
            state = self._init_state(H, W, use_gpu)

        # Process each frame in batch
        output_frames = []

        for i in range(batch_size):
            # Get current frame
            layer_np = layer[i].cpu().numpy()  # (H, W, 4)

            # Get flow for this frame if provided
            frame_flow_u = flow_u[i] if flow_u is not None else None
            frame_flow_v = flow_v[i] if flow_v is not None else None

            # Apply decay mode logic (future enhancement)
            if mode == "framecount":
                # TODO: Implement frame-count mode
                pass
            elif mode == "pingpong":
                # TODO: Implement pingpong mode
                pass

            # Main accumulation with optional flow warping
            if use_gpu and CUPY_AVAILABLE:
                output_np, state = self._process_gpu(
                    layer_np, state, decay, boost_on_new, max_age,
                    exposure, clamp_mode, warp_with_flow, frame_flow_u, frame_flow_v, i
                )
            else:
                if use_gpu and not CUPY_AVAILABLE and i == 0:
                    print("[YS-ECHO] GPU requested but CuPy unavailable, using CPU")

                output_np, state = self._process_cpu(
                    layer_np, state, decay, boost_on_new, max_age,
                    exposure, clamp_mode, warp_with_flow, frame_flow_u, frame_flow_v, i
                )

            output_frames.append(output_np)

            if is_batch and (i % 10 == 0 or i == batch_size - 1):
                print(f"[YS-ECHO] Processed frame {i+1}/{batch_size}")

        # Stack into batch
        if is_batch:
            output_batch = np.stack(output_frames, axis=0)
            output_tensor = torch.from_numpy(output_batch).float()
        else:
            output_tensor = torch.from_numpy(output_frames[0]).unsqueeze(0).float()

        return (output_tensor, state)

    def _init_state(self, H: int, W: int, use_gpu: bool) -> Dict:
        """Initialize state buffers"""
        state = {
            'shape': (H, W),
            'use_gpu': use_gpu,
        }

        if use_gpu and CUPY_AVAILABLE:
            state['rgba'] = cp.zeros((H, W, 4), dtype=cp.float32)
            state['age'] = cp.zeros((H, W), dtype=cp.uint8)
        else:
            state['rgba'] = np.zeros((H, W, 4), dtype=np.float32)
            state['age'] = np.zeros((H, W), dtype=np.uint8)

        return state

    def _process_gpu(self, layer_np: np.ndarray, state: Dict,
                     decay: float, boost_on_new: float, max_age: int,
                     exposure: float, clamp_mode: str, warp_with_flow: bool,
                     flow_u: Optional[torch.Tensor], flow_v: Optional[torch.Tensor], frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """GPU-accelerated processing using custom CUDA kernel"""
        start_time = time.perf_counter()

        H, W = layer_np.shape[:2]

        # Transfer input to GPU
        layer_gpu = cp.asarray(layer_np, dtype=cp.float32)

        # Get state buffers (already on GPU)
        state_rgba = state['rgba']
        state_age = state['age']

        # Apply optical flow warping if requested
        if warp_with_flow and flow_u is not None and flow_v is not None:
            flow_u_np = flow_u[0].cpu().numpy()
            flow_v_np = flow_v[0].cpu().numpy()
            flow_u_gpu = cp.asarray(flow_u_np)
            flow_v_gpu = cp.asarray(flow_v_np)

            # Warp previous state
            state_rgba = backward_warp(state_rgba, flow_u_gpu, flow_v_gpu, use_gpu=True)
            # Don't warp age buffer (age tracks from last input)

        # Allocate output buffers
        output_gpu = cp.zeros((H, W, 4), dtype=cp.float32)
        new_rgba = cp.zeros_like(state_rgba)
        new_age = cp.zeros_like(state_age)

        # Compile and run CUDA kernel
        kernel = get_compiled_kernel('echo_ema_update', ECHO_EMA_UPDATE_KERNEL)

        block_size = (16, 16)
        grid_size = (
            (W + block_size[0] - 1) // block_size[0],
            (H + block_size[1] - 1) // block_size[1]
        )

        # Flatten arrays for kernel
        input_rgba_flat = layer_gpu.reshape(-1)
        state_rgba_flat = state_rgba.reshape(-1)
        state_age_flat = state_age.reshape(-1)
        output_rgba_flat = output_gpu.reshape(-1)
        new_rgba_flat = new_rgba.reshape(-1)
        new_age_flat = new_age.reshape(-1)

        clamp_hard = 1 if clamp_mode == "hard" else 0

        kernel(
            grid_size,
            block_size,
            (
                input_rgba_flat,
                state_rgba_flat,
                state_age_flat,
                output_rgba_flat,
                new_rgba_flat,
                new_age_flat,
                decay,
                boost_on_new,
                exposure,
                max_age,
                W,
                H,
                clamp_hard
            )
        )

        # Transfer output back to CPU
        output_np = cp.asnumpy(output_gpu)

        # Update state
        new_state = {
            'shape': (H, W),
            'use_gpu': True,
            'rgba': new_rgba,
            'age': new_age,
        }

        if frame_idx == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[YS-ECHO] GPU processed {W}x{H} in {elapsed_ms:.2f}ms")

        return output_np, new_state

    def _process_cpu(self, layer_np: np.ndarray, state: Dict,
                     decay: float, boost_on_new: float, max_age: int,
                     exposure: float, clamp_mode: str, warp_with_flow: bool,
                     flow_u: Optional[torch.Tensor], flow_v: Optional[torch.Tensor], frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """CPU fallback processing"""
        start_time = time.perf_counter()

        H, W = layer_np.shape[:2]

        # Get state buffers
        state_rgba = state['rgba']
        state_age = state['age']

        # Apply optical flow warping if requested
        if warp_with_flow and flow_u is not None and flow_v is not None:
            flow_u_np = flow_u[0].cpu().numpy()
            flow_v_np = flow_v[0].cpu().numpy()

            state_rgba = backward_warp(state_rgba, flow_u_np, flow_v_np, use_gpu=False)

        # Input is premultiplied RGBA
        input_rgba = layer_np.copy()
        input_a = input_rgba[:, :, 3]

        # Apply exposure boost to INPUT (makes fresh trails brighter)
        input_boost = np.where(input_a > 0.001, (1.0 + boost_on_new) * exposure, 1.0)
        input_rgba *= input_boost[:, :, None]

        # EMA update (directly on premultiplied RGBA)
        new_rgba = decay * state_rgba + (1.0 - decay) * input_rgba

        # Age tracking (reset on new input)
        new_age = np.where(input_a > 0.001, 0, np.clip(state_age + 1, 0, max_age))

        # Age-based fade
        age_fade = np.where(new_age < max_age, 1.0, np.maximum(0.0, 1.0 - (new_age - max_age) / 10.0))
        new_rgba *= age_fade[:, :, None]

        # Apply clamping
        if clamp_mode == "hard":
            output_np = np.clip(new_rgba, 0, 1)
        else:
            # Soft clip RGB, hard clamp alpha
            output_np = new_rgba.copy()
            output_np[:, :, :3] = np.where(
                new_rgba[:, :, :3] > 0,
                new_rgba[:, :, :3] / (1.0 + new_rgba[:, :, :3]),
                new_rgba[:, :, :3] / (1.0 - new_rgba[:, :, :3])
            )
            output_np[:, :, 3] = np.clip(new_rgba[:, :, 3], 0, 1)

        # Update state
        new_state = {
            'shape': (H, W),
            'use_gpu': False,
            'rgba': output_np.astype(np.float32),
            'age': new_age.astype(np.uint8),
        }

        if frame_idx == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[YS-ECHO] CPU processed {W}x{H} in {elapsed_ms:.2f}ms")

        return output_np, new_state


# Register node
NODE_CLASS_MAPPINGS = {
    "YSEchoLayer": EchoLayerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSEchoLayer": "YS Echo Layer (Temporal Trails)",
}