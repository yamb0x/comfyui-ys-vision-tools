"""
Pixel Sorting Around Tracks Node - Track-Gated Glitch Effect

Apply pixel sorting locally around tracked points or boxes, creating
glitchy/flowy streaks only where tracks exist.

Features:
- Multiple sorting directions: horizontal, vertical, radial, tangent
- Flexible threshold modes: luma, saturation, hue, sobel magnitude
- Region control: circle radius or box dimensions
- GPU-accelerated mask creation and metric computation
- Configurable sorting with ascending/descending/shuffle modes

Author: Yambo Studio
Part of: YS-vision-tools Phase 3 (New Nodes)
"""

import numpy as np
import torch
import time
from typing import Dict, Any, Optional, Tuple

# Try importing GPU libraries
try:
    import cupy as cp
    from ..utils.cuda_kernels import (
        get_compiled_kernel,
        CREATE_TRACK_MASK_KERNEL,
        COMPUTE_METRIC_KERNEL
    )
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class PixelSortingAroundTracksNode:
    """
    Track-gated pixel sorting for localized glitch effects

    Applies pixel sorting (inspired by classical brightness-threshold sorting)
    only in regions around tracked points or boxes. Creates distinctive
    glitchy/flowy visual effects confined to tracked areas.

    Perfect for:
    - Artistic glitch effects on moving objects
    - Data visualization with tracked elements
    - Creative motion-reactive distortions
    - Combining with EchoLayer for persistent streaks
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),  # Source frame (linear float RGB)
                "region_mode": (["radius", "box"], {
                    "default": "radius",
                    "tooltip": "Region shape around tracks"
                }),
                "radius_px": ("FLOAT", {
                    "default": 50.0,
                    "min": 10.0,
                    "max": 300.0,
                    "step": 1.0,
                    "tooltip": "Circle radius in pixels"
                }),
                "box_w": ("FLOAT", {
                    "default": 100.0,
                    "min": 20.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Box width in pixels"
                }),
                "box_h": ("FLOAT", {
                    "default": 100.0,
                    "min": 20.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Box height in pixels"
                }),
                "direction": (["horizontal", "vertical", "radial", "tangent_flow"], {
                    "default": "horizontal",
                    "tooltip": "Sorting direction"
                }),
                "threshold_mode": (["luma", "saturation", "hue", "sobel_mag"], {
                    "default": "luma",
                    "tooltip": "Metric for sorting"
                }),
                "threshold_low": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Lower threshold for sorting"
                }),
                "threshold_high": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Upper threshold for sorting"
                }),
                "sort_order": (["ascending", "descending", "shuffle"], {
                    "default": "ascending",
                    "tooltip": "Sort order for pixels"
                }),
                "segment_min_px": ("INT", {
                    "default": 8,
                    "min": 3,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Minimum segment length for sorting"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Effect opacity"
                }),
                "blend": (["normal", "add", "screen", "lighten"], {
                    "default": "normal",
                    "tooltip": "Blend mode with original"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for shuffle mode"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration for mask/metric (CPU sorting)"
                }),
            },
            "optional": {
                "tracks": ("TRACKS",),
                "boxes": ("BOXES",),
                "mask": ("MASK",),  # Optional external mask
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Effects"

    def execute(self, image: torch.Tensor, region_mode: str, radius_px: float,
                box_w: float, box_h: float, direction: str, threshold_mode: str,
                threshold_low: float, threshold_high: float, sort_order: str,
                segment_min_px: int, opacity: float, blend: str, seed: int,
                use_gpu: bool, tracks: Optional[np.ndarray] = None,
                boxes: Optional[np.ndarray] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,]:
        """
        Apply pixel sorting effect around tracks

        Args:
            image: Input image (BHWC format)
            region_mode: Region shape ('radius' or 'box')
            radius_px: Circle radius
            box_w/box_h: Box dimensions
            direction: Sorting direction
            threshold_mode: Metric for sorting
            threshold_low/high: Threshold range
            sort_order: Sorting order
            segment_min_px: Minimum segment length
            opacity: Effect opacity
            blend: Blend mode
            seed: Random seed
            use_gpu: Use GPU acceleration
            tracks: Track positions (optional)
            boxes: Track boxes (optional)
            mask: External mask (optional)

        Returns:
            Tuple containing processed image
        """
        # Check if batch mode
        batch_size = image.shape[0]
        is_batch = batch_size > 1

        if is_batch:
            print(f"[YS-PIXSORT] BATCH MODE: {batch_size} frames")

        # Process each frame in batch
        output_frames = []

        for frame_idx in range(batch_size):
            # Get current frame
            image_np = image[frame_idx].cpu().numpy()  # (H, W, 3)
            H, W = image_np.shape[:2]

            # Build region mask for this frame
            if mask is not None:
                # Use provided mask
                frame_mask = mask[frame_idx].cpu().numpy() if mask.shape[0] > 1 else mask[0].cpu().numpy()
                mask_np = frame_mask
            elif tracks is not None:
                # Create mask from tracks for this frame
                # Tracks might be per-frame or static
                if isinstance(tracks, list):
                    tracks = np.array(tracks)
                
                if tracks.ndim == 3 and tracks.shape[0] > 1:
                    # Per-frame tracks
                    frame_tracks = tracks[frame_idx]
                else:
                    # Static tracks for all frames
                    frame_tracks = tracks

                mask_np = self._create_track_mask(
                    frame_tracks, H, W, region_mode, radius_px, box_w, box_h, use_gpu, frame_idx
                )
            elif boxes is not None:
                # Create mask from boxes
                if isinstance(boxes, list):
                    boxes = np.array(boxes)
                    
                frame_boxes = boxes[frame_idx] if boxes.ndim == 3 and boxes.shape[0] > 1 else boxes
                track_centers = frame_boxes[:, :2] + frame_boxes[:, 2:] / 2
                mask_np = self._create_track_mask(
                    track_centers, H, W, region_mode, radius_px, box_w, box_h, use_gpu, frame_idx
                )
            else:
                # No mask, process entire image
                mask_np = np.ones((H, W), dtype=np.uint8) * 255

            # Compute sorting metric
            metric_np = self._compute_metric(image_np, threshold_mode, use_gpu, frame_idx)

            # Apply pixel sorting
            np.random.seed(seed + frame_idx)  # Different seed per frame
            sorted_np = self._apply_pixel_sorting(
                image_np, mask_np, metric_np, direction, threshold_low,
                threshold_high, sort_order, segment_min_px
            )

            # Blend with original
            output_np = self._blend_images(image_np, sorted_np, opacity, blend)

            output_frames.append(output_np)

            if is_batch and (frame_idx % 10 == 0 or frame_idx == batch_size - 1):
                print(f"[YS-PIXSORT] Processed frame {frame_idx+1}/{batch_size}")

        # Stack into batch
        if is_batch:
            output_batch = np.stack(output_frames, axis=0)
            output_tensor = torch.from_numpy(output_batch).float()
        else:
            output_tensor = torch.from_numpy(output_frames[0]).unsqueeze(0).float()

        return (output_tensor,)

    def _create_track_mask(self, tracks: np.ndarray, H: int, W: int,
                           region_mode: str, radius: float, box_w: float,
                           box_h: float, use_gpu: bool, frame_idx: int) -> np.ndarray:
        """Create binary mask from track positions"""

        if use_gpu and CUPY_AVAILABLE:
            start_time = time.perf_counter()
            
            # GPU path
            tracks_gpu = cp.asarray(tracks, dtype=cp.float32)
            mask_gpu = cp.zeros((H, W), dtype=cp.uint8)

            kernel = get_compiled_kernel('create_track_mask', CREATE_TRACK_MASK_KERNEL)

            block_size = (16, 16)
            grid_size = (
                (W + block_size[0] - 1) // block_size[0],
                (H + block_size[1] - 1) // block_size[1]
            )

            use_box = 1 if region_mode == "box" else 0

            kernel(
                grid_size,
                block_size,
                (
                    tracks_gpu,
                    len(tracks),
                    mask_gpu,
                    W,
                    H,
                    radius,
                    use_box,
                    box_w,
                    box_h
                )
            )

            mask_np = cp.asnumpy(mask_gpu)
            
            if frame_idx == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[YS-PIXSORT] GPU created mask ({len(tracks)} tracks) @ {W}x{H} in {elapsed_ms:.2f}ms")

        else:
            # CPU path
            if use_gpu and not CUPY_AVAILABLE and frame_idx == 0:
                print("[YS-PIXSORT] GPU requested but CuPy unavailable, using CPU")

            start_time = time.perf_counter()
            
            mask_np = np.zeros((H, W), dtype=np.uint8)

            for track in tracks:
                # Handle different track formats
                if isinstance(track, np.ndarray) and track.ndim == 0:
                    # Scalar - skip
                    continue
                elif isinstance(track, (list, tuple)):
                    tx, ty = float(track[0]), float(track[1])
                elif track.shape == (2,):
                    # 1D array [x, y]
                    tx, ty = float(track[0]), float(track[1])
                elif track.shape == (1, 2):
                    # 2D array [[x, y]]
                    tx, ty = float(track[0, 0]), float(track[0, 1])
                else:
                    if frame_idx == 0:
                        print(f"[YS-PIXSORT] Warning: Unexpected track shape {track.shape}, skipping")
                    continue

                if region_mode == "box":
                    # Box region
                    x_min = int(max(0, tx - box_w / 2))
                    x_max = int(min(W, tx + box_w / 2))
                    y_min = int(max(0, ty - box_h / 2))
                    y_max = int(min(H, ty + box_h / 2))

                    mask_np[y_min:y_max, x_min:x_max] = 255

                else:
                    # Circle region
                    y_coords, x_coords = np.ogrid[:H, :W]
                    dist_sq = (x_coords - tx)**2 + (y_coords - ty)**2
                    circle_mask = dist_sq <= radius**2
                    mask_np[circle_mask] = 255
                    
            if frame_idx == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[YS-PIXSORT] CPU created mask ({len(tracks)} tracks) @ {W}x{H} in {elapsed_ms:.2f}ms")

        return mask_np

    def _compute_metric(self, image: np.ndarray, mode: str, use_gpu: bool, frame_idx: int) -> np.ndarray:
        """Compute sorting metric for image"""

        if use_gpu and CUPY_AVAILABLE:
            start_time = time.perf_counter()
            
            # GPU path
            image_gpu = cp.asarray(image, dtype=cp.float32)
            H, W = image.shape[:2]
            metric_gpu = cp.zeros((H, W), dtype=cp.float32)

            mode_map = {
                'luma': 0,
                'saturation': 1,
                'hue': 2,
                'sobel_mag': 3
            }
            mode_int = mode_map.get(mode, 0)

            kernel = get_compiled_kernel('compute_metric', COMPUTE_METRIC_KERNEL)

            block_size = (16, 16)
            grid_size = (
                (W + block_size[0] - 1) // block_size[0],
                (H + block_size[1] - 1) // block_size[1]
            )

            # Flatten for kernel
            image_flat = image_gpu.reshape(-1)
            metric_flat = metric_gpu.reshape(-1)

            kernel(
                grid_size,
                block_size,
                (
                    image_flat,
                    metric_flat,
                    W,
                    H,
                    mode_int
                )
            )

            metric_np = cp.asnumpy(metric_gpu)
            
            if frame_idx == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[YS-PIXSORT] GPU computed {mode} metric @ {W}x{H} in {elapsed_ms:.2f}ms")

        else:
            start_time = time.perf_counter()
            
            # CPU path
            if mode == 'luma':
                # Rec. 709 luma
                metric_np = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

            elif mode == 'saturation':
                # HSV saturation
                cmax = image.max(axis=2)
                cmin = image.min(axis=2)
                metric_np = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)

            elif mode == 'hue':
                # HSV hue (normalized to [0, 1])
                r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                cmax = image.max(axis=2)
                cmin = image.min(axis=2)
                delta = cmax - cmin

                hue = np.zeros_like(cmax)
                mask = delta > 0

                # Red is max
                r_max = (cmax == r) & mask
                hue[r_max] = ((g[r_max] - b[r_max]) / delta[r_max] % 6) / 6.0

                # Green is max
                g_max = (cmax == g) & mask
                hue[g_max] = ((b[g_max] - r[g_max]) / delta[g_max] + 2) / 6.0

                # Blue is max
                b_max = (cmax == b) & mask
                hue[b_max] = ((r[b_max] - g[b_max]) / delta[b_max] + 4) / 6.0

                metric_np = hue

            elif mode == 'sobel_mag':
                # Sobel gradient magnitude
                from scipy.ndimage import sobel
                luma = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
                gx = sobel(luma, axis=1)
                gy = sobel(luma, axis=0)
                metric_np = np.sqrt(gx**2 + gy**2)

            else:
                metric_np = np.zeros(image.shape[:2])
                
            if frame_idx == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[YS-PIXSORT] CPU computed {mode} metric @ {image.shape[1]}x{image.shape[0]} in {elapsed_ms:.2f}ms")

        return metric_np

    def _apply_pixel_sorting(self, image: np.ndarray, mask: np.ndarray,
                            metric: np.ndarray, direction: str,
                            threshold_low: float, threshold_high: float,
                            sort_order: str, segment_min: int) -> np.ndarray:
        """Apply pixel sorting to image"""

        output = image.copy()
        H, W = image.shape[:2]

        if direction in ["horizontal", "vertical"]:
            # Scanline sorting
            if direction == "horizontal":
                self._sort_scanlines(output, mask, metric, 'h', threshold_low,
                                   threshold_high, sort_order, segment_min)
            else:
                self._sort_scanlines(output, mask, metric, 'v', threshold_low,
                                   threshold_high, sort_order, segment_min)

        elif direction == "radial":
            # TODO: Implement radial sorting
            print("[YS-PIXSORT] Radial sorting not yet implemented, using horizontal")
            self._sort_scanlines(output, mask, metric, 'h', threshold_low,
                               threshold_high, sort_order, segment_min)

        elif direction == "tangent_flow":
            # TODO: Implement tangent flow sorting
            print("[YS-PIXSORT] Tangent flow sorting not yet implemented, using vertical")
            self._sort_scanlines(output, mask, metric, 'v', threshold_low,
                               threshold_high, sort_order, segment_min)

        return output

    def _sort_scanlines(self, image: np.ndarray, mask: np.ndarray,
                       metric: np.ndarray, axis: str, threshold_low: float,
                       threshold_high: float, sort_order: str, segment_min: int):
        """Sort pixels along scanlines (horizontal or vertical)"""

        H, W = image.shape[:2]

        if axis == 'h':
            # Horizontal scanlines
            for y in range(H):
                self._sort_line(
                    image[y, :, :], mask[y, :], metric[y, :],
                    threshold_low, threshold_high, sort_order, segment_min
                )
        else:
            # Vertical scanlines
            for x in range(W):
                self._sort_line(
                    image[:, x, :], mask[:, x], metric[:, x],
                    threshold_low, threshold_high, sort_order, segment_min
                )

    def _sort_line(self, pixels: np.ndarray, line_mask: np.ndarray,
                   line_metric: np.ndarray, threshold_low: float,
                   threshold_high: float, sort_order: str, segment_min: int):
        """Sort pixels in a single line (in-place)"""

        N = len(pixels)

        # Find segments that meet criteria
        # Segment = contiguous run where mask is active AND metric in threshold range
        in_segment = (line_mask > 0) & (line_metric >= threshold_low) & (line_metric <= threshold_high)

        # Find segment boundaries
        segment_starts = []
        segment_ends = []

        i = 0
        while i < N:
            if in_segment[i]:
                start = i
                while i < N and in_segment[i]:
                    i += 1
                end = i

                # Check segment length
                if end - start >= segment_min:
                    segment_starts.append(start)
                    segment_ends.append(end)
            else:
                i += 1

        # Sort each segment
        for start, end in zip(segment_starts, segment_ends):
            segment_pixels = pixels[start:end].copy()
            segment_metrics = line_metric[start:end]

            # Sort based on metric
            if sort_order == "ascending":
                sorted_indices = np.argsort(segment_metrics)
            elif sort_order == "descending":
                sorted_indices = np.argsort(segment_metrics)[::-1]
            else:  # shuffle
                sorted_indices = np.random.permutation(len(segment_metrics))

            # Write sorted pixels back
            pixels[start:end] = segment_pixels[sorted_indices]

    def _blend_images(self, original: np.ndarray, processed: np.ndarray,
                     opacity: float, blend: str) -> np.ndarray:
        """Blend processed image with original"""

        if opacity < 1.0:
            if blend == "normal":
                output = original * (1 - opacity) + processed * opacity
            elif blend == "add":
                output = np.clip(original + processed * opacity, 0, 1)
            elif blend == "screen":
                output = 1 - (1 - original) * (1 - processed * opacity)
            elif blend == "lighten":
                output = np.maximum(original, processed * opacity)
            else:
                output = processed
        else:
            output = processed

        return output


# Register node
NODE_CLASS_MAPPINGS = {
    "YSPixelSortingTracks": PixelSortingAroundTracksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSPixelSortingTracks": "YS Pixel Sorting (Tracks)",
}