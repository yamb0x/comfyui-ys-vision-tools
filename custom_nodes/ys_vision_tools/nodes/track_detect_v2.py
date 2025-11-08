"""
TrackDetect V2 - Advanced feature-based tracking with distinct detection modes
Each mode tracks DIFFERENT visual features for varied artistic effects
GPU-accelerated for 4K@60fps performance
"""

import time
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from scipy.spatial import distance_matrix
from scipy.ndimage import maximum_filter

from ..utils import (
    ensure_numpy_hwc,
    numpy_to_comfyui,
    rgb_to_grayscale,
)

# Try importing GPU dependencies
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class TrackDetectV2Node:
    """
    Advanced tracking with distinct visual feature detection modes.

    Each mode tracks DIFFERENT features:
    - Exploratory Luma: Edge strength (gradient magnitude)
    - Color Hunter: Hue boundaries (color discontinuities)
    - Locked Corners: Harris corners (2D structure)
    - Chroma Density: High saturation regions
    - Phase Congruency: Multi-scale edge features
    """

    # Mode presets (simplified - focus on detection, not physics)
    PRESETS = {
        "Exploratory Luma": {
            "feature_type": "luma_gradient",
            "detection_method": "threshold_adaptive",
            "points_per_frame": 320,
            "min_spacing_px": 8,
            "sensitivity": 0.7,
        },
        "Color Hunter": {
            "feature_type": "hue_contrast",
            "detection_method": "top_k_maxima",
            "points_per_frame": 260,
            "min_spacing_px": 10,
            "sensitivity": 0.6,
        },
        "Locked Corners": {
            "feature_type": "harris_corners",
            "detection_method": "top_k_maxima",
            "points_per_frame": 220,
            "min_spacing_px": 12,
            "sensitivity": 0.5,
        },
        "Chroma Density": {
            "feature_type": "saturation_map",
            "detection_method": "threshold_adaptive",
            "points_per_frame": 280,
            "min_spacing_px": 9,
            "sensitivity": 0.65,
        },
        "Phase Congruency": {
            "feature_type": "multiscale_edges",
            "detection_method": "top_k_maxima",
            "points_per_frame": 240,
            "min_spacing_px": 10,
            "sensitivity": 0.6,
        },
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "tracking_mode": ([
                    "Exploratory Luma",
                    "Color Hunter",
                    "Locked Corners",
                    "Chroma Density",
                    "Phase Congruency",
                ], {
                    "default": "Exploratory Luma",
                    "tooltip": "Each mode tracks different visual features"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (2-5× faster for feature computation)"
                }),
            },
            "optional": {
                # Advanced overrides
                "points_per_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 10, "tooltip": "0 = use preset"}),
                "sensitivity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = use preset"}),

                # Temporal tracking (for video)
                "temporal_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Smooth tracks across frames"}),

                # Internal state (for video loop)
                "prev_state": ("STATE",),
            }
        }

    RETURN_TYPES = ("TRACKS", "STATE", "IMAGE")
    RETURN_NAMES = ("tracks", "state", "debug_viz")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def execute(self, image, tracking_mode, use_gpu, **kwargs):
        """Execute TrackDetect V2 with distinct feature detection"""

        start_time = time.perf_counter()

        # Get config from tracking mode
        config = self._get_config(tracking_mode, **kwargs)

        # Detect batch mode
        is_batch = False
        batch_size = 1

        if torch.is_tensor(image) and len(image.shape) == 4 and image.shape[0] > 1:
            is_batch = True
            batch_size = image.shape[0]
            print(f"[YS-TRACK-V2] BATCH MODE: {batch_size} frames")

        # Process frames
        if is_batch:
            all_tracks = []
            state = kwargs.get('prev_state')
            
            # Get first frame dimensions for adaptive thresholds
            first_frame = ensure_numpy_hwc(image[0:1])
            H, W = first_frame.shape[:2]
            
            # Initialize state once for batch to preserve IDs across frames
            if state is None:
                state = self._init_empty_state()

            for i in range(batch_size):
                frame = image[i:i+1]
                frame_np = ensure_numpy_hwc(frame)

                tracks, state = self._process_single_frame(
                    frame_np, config, state, use_gpu and CUPY_AVAILABLE
                )

                all_tracks.append(tracks)

                if i % 10 == 0 or i == batch_size - 1:
                    print(f"[YS-TRACK-V2] Frame {i+1}/{batch_size}: {len(tracks)} points")

            # Aggregate statistics
            total_tracks = sum(len(t) for t in all_tracks)
            avg_tracks = total_tracks / batch_size if batch_size > 0 else 0
            empty_frames = sum(1 for t in all_tracks if len(t) == 0)
            
            # Debug viz shows LAST frame (most representative of final state)
            viz = self._create_debug_viz(ensure_numpy_hwc(image[-1:]), all_tracks[-1])

            total_time = (time.perf_counter() - start_time) * 1000
            print(f"[YS-TRACK-V2] Batch complete: {batch_size} frames in {total_time:.2f}ms ({total_time/batch_size:.2f}ms/frame)")
            print(f"[YS-TRACK-V2] Average: {avg_tracks:.1f} points/frame, {empty_frames} empty frames")

            return (all_tracks, state, viz)

        # Single frame mode
        image_np = ensure_numpy_hwc(image)
        state = kwargs.get('prev_state')

        tracks, state = self._process_single_frame(
            image_np, config, state, use_gpu and CUPY_AVAILABLE
        )

        viz = self._create_debug_viz(image_np, tracks)

        total_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-TRACK-V2] Single frame: {len(tracks)} points in {total_time:.2f}ms")

        return (tracks, state, viz)

    def _get_config(self, tracking_mode, **kwargs):
        """Get configuration from tracking mode"""

        # Start with preset config
        config = self.PRESETS[tracking_mode].copy()

        print(f"[YS-TRACK-V2] Mode: {tracking_mode}")
        print(f"[YS-TRACK-V2] Feature: {config['feature_type']}, Method: {config['detection_method']}")

        # Apply overrides
        if kwargs.get("points_per_frame", 0) > 0:
            config["points_per_frame"] = kwargs["points_per_frame"]
            print(f"[YS-TRACK-V2] Override points: {config['points_per_frame']}")

        if kwargs.get("sensitivity", 0.0) > 0:
            config["sensitivity"] = kwargs["sensitivity"]
            print(f"[YS-TRACK-V2] Override sensitivity: {config['sensitivity']:.2f}")

        # Temporal smoothing
        config["temporal_smoothing"] = kwargs.get("temporal_smoothing", 0.0)

        return config
    
    def _init_empty_state(self):
        """Initialize empty state for batch processing"""
        return {
            "tracks": np.empty((0, 2), dtype=np.float32),
            "age": np.array([], dtype=np.int32),
            "ids": np.array([], dtype=np.int32),
            "next_id": 0,
        }

    def _process_single_frame(self, image_np, config, prev_state, use_gpu):
        """
        Process single frame through feature detection pipeline.

        Returns: (tracks, state)
        """

        H, W = image_np.shape[:2]

        # ========== STAGE A: Feature Map Computation ==========
        t0 = time.perf_counter()

        feature_map = self._compute_feature_map(
            image_np, config["feature_type"], use_gpu
        )

        t_feature = (time.perf_counter() - t0) * 1000

        # ========== STAGE B: Point Detection ==========
        t0 = time.perf_counter()

        tracks = self._detect_points(
            feature_map,
            config["detection_method"],
            config["points_per_frame"],
            config["min_spacing_px"],
            config["sensitivity"],
            use_gpu
        )

        if len(tracks) == 0:
            print(f"[YS-TRACK-V2] Warning: No points detected")
            return np.empty((0, 2)), prev_state

        t_detect = (time.perf_counter() - t0) * 1000

        # ========== STAGE C: State Management & Temporal Smoothing ==========
        t0 = time.perf_counter()
        
        # Compute resolution-adaptive distance threshold (5% of smaller dimension)
        match_distance = max(50, min(W, H) * 0.05)

        if config["temporal_smoothing"] > 0 and prev_state is not None and "tracks" in prev_state:
            # Apply temporal smoothing
            tracks, state = self._apply_temporal_smoothing(
                tracks, prev_state, config["temporal_smoothing"], match_distance
            )
        else:
            # Update state without smoothing but preserve IDs if possible
            if prev_state is not None and "tracks" in prev_state and len(prev_state["tracks"]) > 0:
                # Match to previous frame to preserve IDs
                state = self._update_state_with_ids(tracks, prev_state, match_distance)
            else:
                # Initialize new state
                next_id = prev_state.get("next_id", 0) if prev_state is not None else 0
                state = {
                    "tracks": tracks,
                    "age": np.zeros(len(tracks), dtype=np.int32),
                    "ids": np.arange(next_id, next_id + len(tracks), dtype=np.int32),
                    "next_id": next_id + len(tracks),
                }

        t_smooth = (time.perf_counter() - t0) * 1000

        print(f"[YS-TRACK-V2] Timings: feature={t_feature:.2f}ms, detect={t_detect:.2f}ms, state={t_smooth:.2f}ms")

        return tracks, state

    # ========== FEATURE MAP COMPUTATION ==========

    def _compute_feature_map(self, image_np, feature_type, use_gpu):
        """
        Compute feature map based on feature type.

        Returns: feature_map (H, W) in [0, 1] range
        """

        if use_gpu and CUPY_AVAILABLE:
            return self._compute_feature_map_gpu(image_np, feature_type)
        else:
            return self._compute_feature_map_cpu(image_np, feature_type)

    def _compute_feature_map_cpu(self, image_np, feature_type):
        """CPU implementation of feature map computation"""

        if feature_type == "luma_gradient":
            # Gradient magnitude (edge strength)
            gray = rgb_to_grayscale(image_np) if len(image_np.shape) == 3 else image_np
            gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            magnitude = np.sqrt(gx**2 + gy**2)
            return self._normalize_percentile(magnitude, 1, 99)

        elif feature_type == "hue_contrast":
            # Hue discontinuities (color boundaries)
            if len(image_np.shape) == 3:
                hsv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
                hue = hsv[:, :, 0] * 360  # [0, 360]

                # Circular gradient (handle wrap at 0/360)
                hue_rad = np.deg2rad(hue)
                hue_x = np.cos(hue_rad)
                hue_y = np.sin(hue_rad)

                gx_x = cv2.Sobel(hue_x, cv2.CV_64F, 1, 0, ksize=3)
                gy_x = cv2.Sobel(hue_x, cv2.CV_64F, 0, 1, ksize=3)
                gx_y = cv2.Sobel(hue_y, cv2.CV_64F, 1, 0, ksize=3)
                gy_y = cv2.Sobel(hue_y, cv2.CV_64F, 0, 1, ksize=3)

                magnitude = np.sqrt(gx_x**2 + gy_x**2 + gx_y**2 + gy_y**2)

                # Weight by saturation (ignore low-saturation areas)
                saturation = hsv[:, :, 1]
                magnitude = magnitude * saturation

                return self._normalize_percentile(magnitude, 1, 99)
            else:
                # Fallback to luma for grayscale
                return self._compute_feature_map_cpu(image_np, "luma_gradient")

        elif feature_type == "harris_corners":
            # Harris corner detector (2D structure)
            gray = rgb_to_grayscale(image_np) if len(image_np.shape) == 3 else image_np

            Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 1.5)
            Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 1.5)
            Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 1.5)

            # Harris corner response: det(M) - k*trace(M)^2
            k = 0.04
            det = Ixx * Iyy - Ixy * Ixy
            trace = Ixx + Iyy
            response = det - k * (trace ** 2)

            # Suppress negative responses
            response = np.maximum(response, 0)

            return self._normalize_percentile(response, 1, 99)

        elif feature_type == "saturation_map":
            # High saturation regions (colorful areas)
            if len(image_np.shape) == 3:
                hsv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
                saturation = hsv[:, :, 1]
                value = hsv[:, :, 2]

                # Saturation weighted by brightness
                chroma = saturation * value

                return self._normalize_percentile(chroma, 5, 99)
            else:
                # Fallback to luma
                return self._compute_feature_map_cpu(image_np, "luma_gradient")

        elif feature_type == "multiscale_edges":
            # Multi-scale edge detection (phase congruency approximation)
            gray = rgb_to_grayscale(image_np) if len(image_np.shape) == 3 else image_np

            # Detect edges at multiple scales
            edges_sum = np.zeros_like(gray)

            for sigma in [1.0, 2.0, 4.0]:
                blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
                gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(gx**2 + gy**2)
                edges_sum += magnitude

            edges_sum /= 3.0  # Average

            return self._normalize_percentile(edges_sum, 1, 99)

        else:
            # Default to luma gradient
            return self._compute_feature_map_cpu(image_np, "luma_gradient")

    def _compute_feature_map_gpu(self, image_np, feature_type):
        """GPU implementation of feature map computation"""

        # Transfer to GPU
        img_gpu = cp.asarray(image_np.astype(np.float32))

        if feature_type == "luma_gradient":
            # Gradient magnitude on GPU using CuPy convolution (no CPU transfer!)
            if len(image_np.shape) == 3:
                gray = 0.299 * img_gpu[:, :, 0] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 2]
            else:
                gray = img_gpu

            # Scharr kernels for GPU convolution
            scharr_x = cp.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=cp.float32)
            scharr_y = cp.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=cp.float32)
            
            # Convolve on GPU
            gx = cp_ndimage.convolve(gray, scharr_x, mode='reflect')
            gy = cp_ndimage.convolve(gray, scharr_y, mode='reflect')

            magnitude = cp.sqrt(gx**2 + gy**2)
            return self._normalize_percentile_gpu(magnitude, 1, 99)

        elif feature_type == "harris_corners":
            # Harris corners on GPU (fully GPU-accelerated, no CPU transfers!)
            if len(image_np.shape) == 3:
                gray = 0.299 * img_gpu[:, :, 0] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 2]
            else:
                gray = img_gpu

            # Sobel kernels for GPU convolution
            sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32) / 8.0
            sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32) / 8.0
            
            # Compute gradients on GPU
            Ix = cp_ndimage.convolve(gray, sobel_x, mode='reflect')
            Iy = cp_ndimage.convolve(gray, sobel_y, mode='reflect')

            # Structure tensor components on GPU
            Ixx = cp_ndimage.gaussian_filter(Ix * Ix, sigma=1.5)
            Iyy = cp_ndimage.gaussian_filter(Iy * Iy, sigma=1.5)
            Ixy = cp_ndimage.gaussian_filter(Ix * Iy, sigma=1.5)

            # Harris corner response on GPU
            k = 0.04
            det = Ixx * Iyy - Ixy * Ixy
            trace = Ixx + Iyy
            response = det - k * (trace ** 2)

            response = cp.maximum(response, 0)

            return self._normalize_percentile_gpu(response, 1, 99)
        
        elif feature_type == "saturation_map":
            # Saturation map on GPU
            if len(image_np.shape) == 3:
                # RGB to HSV conversion on GPU (simplified)
                R = img_gpu[:, :, 0]
                G = img_gpu[:, :, 1]
                B = img_gpu[:, :, 2]
                
                V = cp.maximum(R, cp.maximum(G, B))  # Value
                C = V - cp.minimum(R, cp.minimum(G, B))  # Chroma
                S = cp.where(V > 0, C / (V + 1e-8), 0)  # Saturation
                
                # Saturation weighted by brightness
                chroma = S * V
                
                return self._normalize_percentile_gpu(chroma, 5, 99)
            else:
                # Fallback to luma for grayscale
                return self._compute_feature_map_gpu(image_np, "luma_gradient")
        
        elif feature_type == "multiscale_edges":
            # Multi-scale edge detection on GPU
            if len(image_np.shape) == 3:
                gray = 0.299 * img_gpu[:, :, 0] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 2]
            else:
                gray = img_gpu
            
            # Sobel kernels
            sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32) / 8.0
            sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32) / 8.0
            
            # Detect edges at multiple scales on GPU
            edges_sum = cp.zeros_like(gray)
            
            for sigma in [1.0, 2.0, 4.0]:
                blurred = cp_ndimage.gaussian_filter(gray, sigma=sigma)
                gx = cp_ndimage.convolve(blurred, sobel_x, mode='reflect')
                gy = cp_ndimage.convolve(blurred, sobel_y, mode='reflect')
                magnitude = cp.sqrt(gx**2 + gy**2)
                edges_sum += magnitude
            
            edges_sum /= 3.0  # Average
            
            return self._normalize_percentile_gpu(edges_sum, 1, 99)

        else:
            # Fallback to CPU for complex feature types (e.g., hue_contrast)
            return self._compute_feature_map_cpu(image_np, feature_type)

    # ========== POINT DETECTION ==========

    def _detect_points(self, feature_map, detection_method, n_points, min_spacing, sensitivity, use_gpu):
        """
        Detect points from feature map.

        Returns: tracks (N, 2) array of (x, y) coordinates
        """

        # Convert to numpy if GPU array
        if use_gpu and CUPY_AVAILABLE and isinstance(feature_map, cp.ndarray):
            feature_np = cp.asnumpy(feature_map)
        else:
            feature_np = feature_map

        H, W = feature_np.shape[:2]

        if detection_method == "top_k_maxima":
            # Find local maxima and take top K
            tracks = self._detect_top_k_maxima(feature_np, n_points, min_spacing, sensitivity)

        elif detection_method == "threshold_adaptive":
            # Adaptive thresholding with non-maximum suppression
            tracks = self._detect_threshold_adaptive(feature_np, n_points, min_spacing, sensitivity)

        else:
            # Default to top-K
            tracks = self._detect_top_k_maxima(feature_np, n_points, min_spacing, sensitivity)

        return tracks

    def _detect_top_k_maxima(self, feature_map, n_points, min_spacing, sensitivity):
        """Detect top-K local maxima"""

        H, W = feature_map.shape[:2]

        # Local maxima detection with non-maximum suppression
        local_max = maximum_filter(feature_map, size=int(min_spacing))
        is_maxima = (feature_map == local_max)

        # Apply sensitivity threshold
        threshold = np.percentile(feature_map, (1 - sensitivity) * 100)
        is_maxima = is_maxima & (feature_map > threshold)

        # Get coordinates
        coords = np.argwhere(is_maxima)

        if len(coords) == 0:
            return np.empty((0, 2))

        # Get feature values
        values = feature_map[coords[:, 0], coords[:, 1]]

        # Sort by value (descending)
        sorted_idx = np.argsort(values)[::-1]

        # Take top K
        top_k = min(n_points, len(sorted_idx))
        seeds = coords[sorted_idx[:top_k]]

        # Convert (y, x) to (x, y)
        tracks = seeds[:, [1, 0]].astype(np.float32)

        return tracks

    def _detect_threshold_adaptive(self, feature_map, n_points, min_spacing, sensitivity):
        """Adaptive threshold detection with spacing"""

        H, W = feature_map.shape[:2]

        # Adaptive threshold
        threshold = np.percentile(feature_map, (1 - sensitivity) * 100)
        mask = feature_map > threshold

        # Get all candidate points
        coords = np.argwhere(mask)

        if len(coords) == 0:
            return np.empty((0, 2))

        # Get feature values
        values = feature_map[coords[:, 0], coords[:, 1]]

        # Sort by value (descending)
        sorted_idx = np.argsort(values)[::-1]
        coords = coords[sorted_idx]
        values = values[sorted_idx]

        # Greedy selection with minimum spacing
        selected = []
        selected_coords = []

        for i in range(len(coords)):
            if len(selected) >= n_points:
                break

            coord = coords[i]

            # Check spacing with already selected points
            if len(selected_coords) == 0:
                selected.append(coord)
                selected_coords.append(coord)
            else:
                selected_arr = np.array(selected_coords)
                distances = np.sqrt(np.sum((selected_arr - coord)**2, axis=1))

                if np.min(distances) >= min_spacing:
                    selected.append(coord)
                    selected_coords.append(coord)

        if len(selected) == 0:
            return np.empty((0, 2))

        selected = np.array(selected)

        # Convert (y, x) to (x, y)
        tracks = selected[:, [1, 0]].astype(np.float32)

        return tracks

    # ========== TEMPORAL SMOOTHING ==========
    
    def _update_state_with_ids(self, tracks, prev_state, match_distance):
        """Update state with ID preservation but no smoothing"""
        
        prev_tracks = prev_state["tracks"]
        prev_age = prev_state["age"]
        prev_ids = prev_state["ids"]
        next_id = prev_state.get("next_id", prev_ids.max() + 1 if len(prev_ids) > 0 else 0)
        
        if len(prev_tracks) == 0:
            # No previous tracks
            return {
                "tracks": tracks,
                "age": np.zeros(len(tracks), dtype=np.int32),
                "ids": np.arange(next_id, next_id + len(tracks), dtype=np.int32),
                "next_id": next_id + len(tracks),
            }
        
        # Match current tracks to previous
        dist_mat = distance_matrix(tracks, prev_tracks)
        matches = np.argmin(dist_mat, axis=1)
        distances = np.min(dist_mat, axis=1)
        
        new_tracks = tracks.copy()
        new_age = np.zeros(len(tracks), dtype=np.int32)
        new_ids = np.zeros(len(tracks), dtype=np.int32)
        
        new_id_counter = next_id
        
        for i in range(len(tracks)):
            matched_idx = matches[i]
            dist = distances[i]
            
            if dist < match_distance:
                # Matched to previous track - preserve ID
                new_age[i] = prev_age[matched_idx] + 1
                new_ids[i] = prev_ids[matched_idx]
            else:
                # New track - assign new ID
                new_age[i] = 0
                new_ids[i] = new_id_counter
                new_id_counter += 1
        
        return {
            "tracks": new_tracks,
            "age": new_age,
            "ids": new_ids,
            "next_id": new_id_counter,
        }

    def _apply_temporal_smoothing(self, tracks, prev_state, smoothing_factor, match_distance):
        """Apply temporal smoothing for video stability"""

        prev_tracks = prev_state["tracks"]
        prev_age = prev_state["age"]
        prev_ids = prev_state["ids"]

        # Match current tracks with previous tracks (nearest neighbor)
        if len(prev_tracks) == 0:
            # No previous tracks, initialize state
            state = {
                "tracks": tracks,
                "age": np.zeros(len(tracks)),
                "ids": np.arange(len(tracks)),
            }
            return tracks, state

        # Compute distance matrix
        dist_mat = distance_matrix(tracks, prev_tracks)

        # Simple greedy matching (TODO: Hungarian algorithm for optimal)
        n_current = len(tracks)
        n_prev = len(prev_tracks)

        if n_current <= n_prev:
            # Match each current to closest previous
            matches = np.argmin(dist_mat, axis=1)
            distances = np.min(dist_mat, axis=1)

            # Smooth matched tracks
            smoothed_tracks = np.zeros_like(tracks)
            new_age = np.zeros(n_current)
            new_ids = np.zeros(n_current, dtype=int)

            for i in range(n_current):
                matched_idx = matches[i]
                dist = distances[i]

                # Smooth if distance is reasonable (not a new track)
                if dist < match_distance:  # Max distance threshold
                    smoothed_tracks[i] = smoothing_factor * prev_tracks[matched_idx] + (1 - smoothing_factor) * tracks[i]
                    new_age[i] = prev_age[matched_idx] + 1
                    new_ids[i] = prev_ids[matched_idx]
                else:
                    # New track
                    smoothed_tracks[i] = tracks[i]
                    new_age[i] = 0
                    new_ids[i] = prev_ids.max() + 1 + i

        else:
            # More current than previous, some are new
            matches = np.argmin(dist_mat, axis=1)
            distances = np.min(dist_mat, axis=1)

            smoothed_tracks = np.zeros_like(tracks)
            new_age = np.zeros(n_current)
            new_ids = np.zeros(n_current, dtype=int)

            max_id = prev_ids.max() if len(prev_ids) > 0 else 0

            for i in range(n_current):
                matched_idx = matches[i]
                dist = distances[i]

                if dist < match_distance:
                    smoothed_tracks[i] = smoothing_factor * prev_tracks[matched_idx] + (1 - smoothing_factor) * tracks[i]
                    new_age[i] = prev_age[matched_idx] + 1
                    new_ids[i] = prev_ids[matched_idx]
                else:
                    smoothed_tracks[i] = tracks[i]
                    new_age[i] = 0
                    new_ids[i] = max_id + 1 + i

        state = {
            "tracks": smoothed_tracks,
            "age": new_age,
            "ids": new_ids,
        }

        return smoothed_tracks, state

    # ========== UTILITY FUNCTIONS ==========

    def _normalize_percentile(self, arr, low, high):
        """Normalize array to [0, 1] using percentile clipping"""

        p_low = np.percentile(arr, low)
        p_high = np.percentile(arr, high)

        clipped = np.clip(arr, p_low, p_high)

        if p_high > p_low:
            normalized = (clipped - p_low) / (p_high - p_low)
        else:
            normalized = np.zeros_like(clipped)

        return normalized.astype(np.float32)

    def _normalize_percentile_gpu(self, arr, low, high):
        """GPU version of percentile normalization"""

        p_low = float(cp.percentile(arr, low))
        p_high = float(cp.percentile(arr, high))

        clipped = cp.clip(arr, p_low, p_high)

        if p_high > p_low:
            normalized = (clipped - p_low) / (p_high - p_low)
        else:
            normalized = cp.zeros_like(clipped)

        return normalized.astype(cp.float32)

    def _create_debug_viz(self, image_np, tracks):
        """Create debug visualization with tracks"""

        # Convert to uint8
        if image_np.max() <= 1.0:
            viz = (image_np * 255).astype(np.uint8)
        else:
            viz = image_np.astype(np.uint8)

        # Ensure RGB
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)
        elif viz.shape[2] == 4:
            viz = viz[:, :, :3]

        # Draw tracks
        for x, y in tracks.astype(int):
            if 0 <= x < viz.shape[1] and 0 <= y < viz.shape[0]:
                cv2.circle(viz, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.circle(viz, (int(x), int(y)), 5, (255, 255, 255), 1)

        # Convert to ComfyUI format
        viz_float = viz.astype(np.float32) / 255.0
        return numpy_to_comfyui(viz_float)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "YS_TrackDetectV2": TrackDetectV2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_TrackDetectV2": "2D Tracker (Colors/Luma) 🎨"
}