"""
Track Jitter Node for YS-vision-tools
Applies jitter and push-apart effects to track points

GPU-ACCELERATED with vectorized distance calculations:
- Push Apart: Iteratively separates overlapping points
- Jitter: Consistent random offsets per point (stable across frames)
- Fully GPU-accelerated distance matrix computation
- Handles batch video processing
"""

import numpy as np
import time
from typing import Dict, Any, Tuple

# GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class TrackJitterNode:
    """
    Apply jitter and push-apart effects to tracked points

    Features:
    - Push Apart: Prevents point overlap by spreading clusters
    - Jitter: Adds consistent random variation per point
    - GPU-accelerated distance calculations
    - Deterministic per-point offsets (stable across frames)
    - Batch processing for video
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "push_apart": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Minimum distance between points. Iteratively pushes overlapping points apart."
                }),
                "jitter_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Random variation applied to each point. Consistent across frames."
                }),
                "iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of push-apart iterations. More = better separation but slower."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Random seed for jitter. Same seed = same jitter pattern."
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration for distance calculations (10-50× faster)"
                }),
            }
        }

    RETURN_TYPES = ("TRACKS",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def __init__(self):
        """Initialize jitter node"""
        self.jitter_cache = {}  # Cache jitter offsets per point

    def execute(self, tracks, push_apart, jitter_amount, iterations, seed, use_gpu):
        """Apply jitter and push-apart to tracks"""

        print(f"\n[YS-JITTER] Executing TrackJitter")
        print(f"[YS-JITTER] Push apart: {push_apart}, Jitter: {jitter_amount}")
        print(f"[YS-JITTER] Tracks type: {type(tracks)}")

        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            batch_size = len(tracks)
            print(f"[YS-JITTER] BATCH MODE: {batch_size} frames")

            batch_tracks = []
            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                jittered = self._process_single_frame(
                    frame_tracks, push_apart, jitter_amount,
                    iterations, seed, use_gpu
                )
                batch_tracks.append(jittered)

                # Progress logging
                if i % 10 == 0 or i == batch_size - 1:
                    print(f"[YS-JITTER] Processed frame {i+1}/{batch_size}")

            print(f"[YS-JITTER] Returning batch: {len(batch_tracks)} frames")
            return (batch_tracks,)

        # Single frame mode
        print(f"[YS-JITTER] SINGLE MODE")
        jittered = self._process_single_frame(
            tracks, push_apart, jitter_amount,
            iterations, seed, use_gpu
        )
        return (jittered,)

    def _process_single_frame(self, tracks, push_apart, jitter_amount,
                             iterations, seed, use_gpu):
        """Process single frame with GPU/CPU path selection"""

        # Convert to numpy array
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks)

        if len(tracks) == 0:
            return tracks

        # Make a copy to avoid modifying input
        tracks = tracks.copy().astype(np.float32)

        # Apply jitter first (deterministic per point)
        if jitter_amount > 0:
            tracks = self._apply_jitter(tracks, jitter_amount, seed)

        # Apply push-apart iterations
        if push_apart > 0:
            if use_gpu and CUPY_AVAILABLE:
                try:
                    tracks = self._push_apart_gpu(tracks, push_apart, iterations)
                except Exception as e:
                    error_msg = str(e)
                    # Provide helpful message for common CUDA compilation errors
                    if "cl.exe" in error_msg or "nvcc" in error_msg:
                        print(f"[YS-JITTER] GPU compilation unavailable (missing MSVC compiler), using CPU fallback")
                        print(f"[YS-JITTER] Note: Install 'Desktop development with C++' from Visual Studio for GPU support")
                    else:
                        print(f"[YS-JITTER] GPU push-apart failed: {e}")
                    print(f"[YS-JITTER] Falling back to CPU")
                    tracks = self._push_apart_cpu(tracks, push_apart, iterations)
            else:
                if use_gpu and not CUPY_AVAILABLE:
                    print(f"[YS-JITTER] GPU requested but CuPy unavailable, using CPU")
                tracks = self._push_apart_cpu(tracks, push_apart, iterations)

        return tracks

    def _apply_jitter(self, tracks: np.ndarray, jitter_amount: float, seed: int) -> np.ndarray:
        """
        Apply consistent random jitter to each point.
        Uses hash of point index for deterministic per-point offsets.
        """
        rng = np.random.RandomState(seed)

        # Generate jitter offsets for each point
        # Use point index as additional seed factor for determinism
        n_points = len(tracks)

        # Create per-point jitter (consistent across frames)
        angles = rng.uniform(0, 2 * np.pi, n_points)
        magnitudes = rng.uniform(0, jitter_amount, n_points)

        # Convert polar to cartesian
        jitter_x = magnitudes * np.cos(angles)
        jitter_y = magnitudes * np.sin(angles)

        tracks[:, 0] += jitter_x
        tracks[:, 1] += jitter_y

        return tracks

    def _push_apart_gpu(self, tracks: np.ndarray, min_distance: float,
                       iterations: int) -> np.ndarray:
        """
        GPU-accelerated push-apart using vectorized distance calculations.
        Iteratively pushes overlapping points away from each other.
        """
        start_time = time.perf_counter()

        # Transfer to GPU
        points_gpu = cp.asarray(tracks, dtype=cp.float32)
        n_points = len(points_gpu)
        min_dist_sq = min_distance * min_distance

        for iteration in range(iterations):
            # Compute pairwise distance matrix on GPU
            # Shape: (N, 1, 2) - (1, N, 2) = (N, N, 2)
            diff = points_gpu[:, cp.newaxis, :] - points_gpu[cp.newaxis, :, :]

            # Distance squared for each pair
            dist_sq = cp.sum(diff ** 2, axis=2)

            # Compute normalized repulsion vectors
            dist = cp.sqrt(dist_sq + 1e-6)  # Avoid division by zero

            # Mask: distances below threshold (but not self)
            # Use simple comparison instead of boolean mask to avoid kernel compilation
            mask = (dist_sq < min_dist_sq) & (dist_sq > 1e-6)

            # Check if any overlapping points exist (use sum instead of any to avoid compilation)
            num_overlaps = float(cp.sum(mask))
            if num_overlaps == 0:
                # No more overlapping points
                break

            # Repulsion force: inversely proportional to distance
            # Force stronger when points are closer
            # Use masking via multiplication instead of cp.where to avoid compilation
            force_magnitude = ((min_distance - dist) / (dist + 1e-6)) * mask

            # Direction vectors (normalized difference)
            direction = diff / (dist[:, :, cp.newaxis] + 1e-6)

            # Apply repulsion forces
            forces = direction * force_magnitude[:, :, cp.newaxis]

            # Sum all forces acting on each point
            displacement = cp.sum(forces, axis=1)

            # Apply displacement with damping
            damping = 0.5  # Prevents oscillation
            points_gpu += displacement * damping

        # Transfer back to CPU
        result = cp.asnumpy(points_gpu)

        gpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-JITTER] GPU push-apart {n_points} points ({iterations} iterations) in {gpu_time:.2f}ms")

        return result

    def _push_apart_cpu(self, tracks: np.ndarray, min_distance: float,
                       iterations: int) -> np.ndarray:
        """
        CPU fallback for push-apart.
        Same algorithm as GPU version but using NumPy.
        """
        start_time = time.perf_counter()

        points = tracks.astype(np.float32)
        n_points = len(points)
        min_dist_sq = min_distance * min_distance

        for iteration in range(iterations):
            # Compute pairwise distance matrix
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)

            # Mask: distances below threshold (but not self)
            mask = (dist_sq < min_dist_sq) & (dist_sq > 1e-6)

            if not np.any(mask):
                break

            # Compute normalized repulsion vectors
            dist = np.sqrt(dist_sq + 1e-6)

            # Repulsion force
            force_magnitude = np.where(
                mask,
                (min_distance - dist) / (dist + 1e-6),
                0.0
            )

            # Direction vectors
            direction = diff / (dist[:, :, np.newaxis] + 1e-6)

            # Apply repulsion forces
            forces = direction * force_magnitude[:, :, np.newaxis]
            displacement = np.sum(forces, axis=1)

            # Apply with damping
            damping = 0.5
            points += displacement * damping

        cpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-JITTER] CPU push-apart {n_points} points ({iterations} iterations) in {cpu_time:.2f}ms")

        return points


# Node registration
NODE_CLASS_MAPPINGS = {
    "YSTrackJitter": TrackJitterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSTrackJitter": "YS Track Jitter"
}
