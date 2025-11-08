"""
Optical Flow Warping Utilities

GPU-accelerated backward warping for temporal effects.
Used by EchoLayerNode for motion-aware trail effects.

Author: Yambo Studio
"""

import numpy as np
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def backward_warp_gpu(image_gpu, flow_u_gpu, flow_v_gpu):
    """
    GPU-accelerated backward warping using optical flow

    Warps image according to flow field using bilinear interpolation.
    For each output pixel (x, y), samples from input at (x - u, y - v).

    Args:
        image_gpu: CuPy array (H, W) or (H, W, C)
        flow_u_gpu: CuPy array (H, W) - horizontal flow
        flow_v_gpu: CuPy array (H, W) - vertical flow

    Returns:
        Warped image as CuPy array (same shape as input)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU warping")

    H, W = image_gpu.shape[:2]
    is_multichannel = image_gpu.ndim == 3

    # Create coordinate grids
    y_coords, x_coords = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')

    # Apply backward flow (sample from source position)
    src_x = x_coords - flow_u_gpu
    src_y = y_coords - flow_v_gpu

    # Bilinear interpolation using map_coordinates
    from cupyx.scipy.ndimage import map_coordinates

    if is_multichannel:
        C = image_gpu.shape[2]
        warped = cp.zeros_like(image_gpu)

        for c in range(C):
            coords = cp.stack([src_y, src_x], axis=0)
            warped[:, :, c] = map_coordinates(
                image_gpu[:, :, c],
                coords,
                order=1,  # Bilinear
                mode='constant',
                cval=0.0
            )
    else:
        coords = cp.stack([src_y, src_x], axis=0)
        warped = map_coordinates(
            image_gpu,
            coords,
            order=1,
            mode='constant',
            cval=0.0
        )

    return warped


def backward_warp_cpu(image, flow_u, flow_v):
    """
    CPU fallback for backward warping using optical flow

    Args:
        image: NumPy array (H, W) or (H, W, C)
        flow_u: NumPy array (H, W) - horizontal flow
        flow_v: NumPy array (H, W) - vertical flow

    Returns:
        Warped image as NumPy array
    """
    import cv2

    H, W = image.shape[:2]

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Apply backward flow
    map_x = (x_coords - flow_u).astype(np.float32)
    map_y = (y_coords - flow_v).astype(np.float32)

    # Remap using OpenCV
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped


def backward_warp(image, flow_u, flow_v, use_gpu=True):
    """
    Backward warp with automatic GPU/CPU dispatch

    Args:
        image: Image array (H, W) or (H, W, C)
        flow_u: Horizontal flow field (H, W)
        flow_v: Vertical flow field (H, W)
        use_gpu: Use GPU if available

    Returns:
        Warped image (same type as input)
    """
    start_time = time.perf_counter()

    if use_gpu and CUPY_AVAILABLE:
        # GPU path
        is_cupy_input = isinstance(image, cp.ndarray)

        if not is_cupy_input:
            image_gpu = cp.asarray(image)
            flow_u_gpu = cp.asarray(flow_u)
            flow_v_gpu = cp.asarray(flow_v)
        else:
            image_gpu = image
            flow_u_gpu = flow_u
            flow_v_gpu = flow_v

        warped_gpu = backward_warp_gpu(image_gpu, flow_u_gpu, flow_v_gpu)

        if not is_cupy_input:
            warped = cp.asnumpy(warped_gpu)
        else:
            warped = warped_gpu

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-WARP] GPU backward warp {image.shape} in {elapsed_ms:.2f}ms")

    else:
        # CPU fallback
        if use_gpu and not CUPY_AVAILABLE:
            print("[YS-WARP] GPU requested but CuPy unavailable, using CPU")

        if isinstance(image, cp.ndarray):
            image = cp.asnumpy(image)
            flow_u = cp.asnumpy(flow_u)
            flow_v = cp.asnumpy(flow_v)

        warped = backward_warp_cpu(image, flow_u, flow_v)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-WARP] CPU backward warp {image.shape} in {elapsed_ms:.2f}ms")

    return warped


def estimate_optical_flow_gpu(prev_frame, curr_frame, method='farneback'):
    """
    Estimate optical flow between two frames (GPU-accelerated)

    Args:
        prev_frame: Previous frame (H, W, 3) RGB float [0, 1]
        curr_frame: Current frame (H, W, 3) RGB float [0, 1]
        method: Flow estimation method ('farneback' only for now)

    Returns:
        Tuple of (flow_u, flow_v) as CuPy or NumPy arrays
    """
    # Convert to grayscale
    if prev_frame.ndim == 3:
        # RGB to grayscale
        prev_gray = 0.2126 * prev_frame[:, :, 0] + 0.7152 * prev_frame[:, :, 1] + 0.0722 * prev_frame[:, :, 2]
        curr_gray = 0.2126 * curr_frame[:, :, 0] + 0.7152 * curr_frame[:, :, 1] + 0.0722 * curr_frame[:, :, 2]
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame

    # For now, use OpenCV CPU implementation
    # TODO: Implement GPU optical flow using CuPy or CUDA
    import cv2

    if isinstance(prev_gray, cp.ndarray):
        prev_gray = cp.asnumpy(prev_gray)
        curr_gray = cp.asnumpy(curr_gray)

    # Convert to uint8 for OpenCV
    prev_uint8 = (np.clip(prev_gray, 0, 1) * 255).astype(np.uint8)
    curr_uint8 = (np.clip(curr_gray, 0, 1) * 255).astype(np.uint8)

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_uint8,
        curr_uint8,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]

    return flow_u, flow_v
