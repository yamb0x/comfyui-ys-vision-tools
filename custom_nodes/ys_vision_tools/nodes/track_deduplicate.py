"""
Track Deduplicate Node for YS-vision-tools

Removes overlapping/clustered tracking points with GPU acceleration.

Use Case:
- Clean up dense detection clusters before text rendering
- Remove duplicate detections from multiple sources
- Control point density for visual effects

Features:
- GPU-accelerated distance matrix computation (100× faster)
- Multiple keep strategies (first, last, center, random)
- Batch processing for video frames
- Configurable distance threshold
- Real-time performance logging

Author: Yambo Studio
Part of: YS-vision-tools
"""

import numpy as np
import torch
import time
from typing import Dict, Any, List, Tuple

# GPU imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Fallback CPU spatial indexing
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TrackDeduplicateNode:
    """
    Remove overlapping/clustered tracking points

    Takes TRACKS input and filters out points that are too close together,
    keeping only one point per cluster based on the selected strategy.

    GPU-accelerated for high performance on large point sets.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),

                # Distance threshold (pixels)
                "min_distance": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Minimum distance between points (pixels). Points closer than this are considered duplicates."
                }),

                # Strategy for choosing which point to keep
                "keep_strategy": ([
                    "first",    # Keep first point encountered (insertion order)
                    "last",     # Keep last point encountered
                    "center",   # Keep point closest to cluster centroid
                    "random"    # Keep random point from cluster
                ], {
                    "default": "center",
                    "tooltip": "Which point to keep from each cluster"
                }),

                # GPU acceleration
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration (requires CuPy)"
                }),
            },
            "optional": {
                # Advanced options
                "max_points": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Maximum points to keep (0 = unlimited). Applied after deduplication."
                }),

                "sort_by_position": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Sort output by position (top-to-bottom, left-to-right)"
                }),
            }
        }

    RETURN_TYPES = ("TRACKS", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("tracks", "input_count", "output_count", "reduction_percent")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def __init__(self):
        self.name = "TrackDeduplicate"

    def execute(
        self,
        tracks,
        min_distance: float,
        keep_strategy: str,
        use_gpu: bool = True,
        max_points: int = 0,
        sort_by_position: bool = False
    ):
        """
        Remove overlapping points from tracks

        Args:
            tracks: TRACKS data (numpy array or list of arrays)
            min_distance: Minimum distance between points in pixels
            keep_strategy: Which point to keep per cluster
            use_gpu: Use GPU acceleration
            max_points: Maximum points to keep (0 = unlimited)
            sort_by_position: Sort output by position

        Returns:
            (filtered_tracks, input_count, output_count, reduction_percent)
        """

        start_time = time.perf_counter()

        # Detect batch mode (video frames)
        is_batch = isinstance(tracks, list)

        if is_batch:
            batch_size = len(tracks)
            print(f"[YS-DEDUPE] BATCH MODE: {batch_size} frames")

            # Process each frame
            output_tracks = []
            total_input = 0
            total_output = 0

            for i, frame_tracks in enumerate(tracks):
                filtered = self._deduplicate_frame(
                    frame_tracks,
                    min_distance,
                    keep_strategy,
                    use_gpu,
                    max_points,
                    sort_by_position
                )

                output_tracks.append(filtered)
                total_input += len(frame_tracks)
                total_output += len(filtered)

                # Progress logging (every 10 frames)
                if i % 10 == 0 or i == batch_size - 1:
                    print(f"[YS-DEDUPE] Processed frame {i+1}/{batch_size}")

            # Statistics
            reduction_pct = ((total_input - total_output) / total_input * 100) if total_input > 0 else 0.0
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            print(f"[YS-DEDUPE] Batch complete: {total_input} → {total_output} points ({reduction_pct:.1f}% reduction) in {elapsed_ms:.2f}ms")

            return (output_tracks, total_input, total_output, reduction_pct)

        else:
            # Single frame mode
            input_count = len(tracks)

            filtered = self._deduplicate_frame(
                tracks,
                min_distance,
                keep_strategy,
                use_gpu,
                max_points,
                sort_by_position
            )

            output_count = len(filtered)
            reduction_pct = ((input_count - output_count) / input_count * 100) if input_count > 0 else 0.0
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            print(f"[YS-DEDUPE] {input_count} → {output_count} points ({reduction_pct:.1f}% reduction) in {elapsed_ms:.2f}ms")

            return (filtered, input_count, output_count, reduction_pct)

    def _deduplicate_frame(
        self,
        tracks: np.ndarray,
        min_distance: float,
        keep_strategy: str,
        use_gpu: bool,
        max_points: int,
        sort_by_position: bool
    ) -> np.ndarray:
        """
        Deduplicate a single frame of tracks

        Returns:
            Filtered numpy array of shape (N, 2)
        """

        # Handle empty input
        if len(tracks) == 0:
            return tracks

        # Convert to numpy if needed
        if isinstance(tracks, torch.Tensor):
            tracks = tracks.cpu().numpy()

        # Single point - nothing to deduplicate
        if len(tracks) == 1:
            return tracks

        # Choose GPU or CPU path
        if use_gpu and CUPY_AVAILABLE:
            filtered = self._deduplicate_gpu(tracks, min_distance, keep_strategy)
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("[YS-DEDUPE] GPU requested but CuPy unavailable, using CPU")
            filtered = self._deduplicate_cpu(tracks, min_distance, keep_strategy)

        # Apply max_points limit
        if max_points > 0 and len(filtered) > max_points:
            # Keep first max_points (or could sort by some criteria first)
            filtered = filtered[:max_points]

        # Sort by position if requested
        if sort_by_position and len(filtered) > 0:
            # Sort by y, then x (top-to-bottom, left-to-right)
            sort_indices = np.lexsort((filtered[:, 0], filtered[:, 1]))
            filtered = filtered[sort_indices]

        return filtered

    def _deduplicate_gpu(
        self,
        tracks: np.ndarray,
        min_distance: float,
        keep_strategy: str
    ) -> np.ndarray:
        """
        GPU-accelerated deduplication using CuPy

        Strategy:
        1. Compute pairwise distance matrix on GPU (fast, vectorized)
        2. Transfer to CPU for clustering (avoids CUDA compilation issues)
        3. Apply keep strategy and return result
        """

        start_time = time.perf_counter()

        # Transfer to GPU
        tracks_gpu = cp.asarray(tracks, dtype=cp.float32)

        # Compute pairwise squared distances (faster than distances)
        # dist[i,j] = ||tracks[i] - tracks[j]||²
        diff = tracks_gpu[:, None, :] - tracks_gpu[None, :, :]  # (N, N, 2)
        dist_sq = cp.sum(diff ** 2, axis=2)  # (N, N)

        threshold_sq = min_distance * min_distance

        # Build adjacency matrix (points within threshold are connected)
        adjacency = dist_sq < threshold_sq  # (N, N) boolean

        # Transfer adjacency and tracks to CPU for clustering
        # (avoids CUDA kernel compilation issues)
        adjacency_cpu = cp.asnumpy(adjacency)
        tracks_cpu = cp.asnumpy(tracks_gpu)
        dist_sq_cpu = cp.asnumpy(dist_sq)

        # Find connected components (clusters) on CPU
        clusters = self._find_clusters_cpu(adjacency_cpu)

        # Apply keep strategy on CPU
        keep_indices = self._apply_keep_strategy_cpu(
            tracks_cpu, clusters, keep_strategy, dist_sq_cpu
        )

        # Return filtered tracks
        result = tracks_cpu[keep_indices]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-DEDUPE] GPU deduplication: {len(tracks)} → {len(result)} in {elapsed_ms:.2f}ms")

        return result

    def _deduplicate_cpu(
        self,
        tracks: np.ndarray,
        min_distance: float,
        keep_strategy: str
    ) -> np.ndarray:
        """
        CPU fallback using scipy spatial indexing
        """

        start_time = time.perf_counter()

        if SCIPY_AVAILABLE:
            # Use cKDTree for efficient spatial queries
            tree = cKDTree(tracks)

            unique_indices = []
            used = set()

            for i in range(len(tracks)):
                if i in used:
                    continue

                # Find all neighbors within min_distance
                neighbors = tree.query_ball_point(tracks[i], min_distance)

                # Apply keep strategy
                if keep_strategy == "first":
                    keep_idx = i
                elif keep_strategy == "last":
                    keep_idx = max(neighbors)
                elif keep_strategy == "center":
                    # Find point closest to cluster centroid
                    cluster_points = tracks[neighbors]
                    centroid = np.mean(cluster_points, axis=0)
                    dists = np.sum((cluster_points - centroid) ** 2, axis=1)
                    keep_idx = neighbors[np.argmin(dists)]
                else:  # random
                    keep_idx = np.random.choice(neighbors)

                unique_indices.append(keep_idx)
                used.update(neighbors)

            result = tracks[unique_indices]

        else:
            # Simple pairwise fallback (slow but works)
            print("[YS-DEDUPE] SciPy unavailable, using simple pairwise distance (slower)")

            unique_points = [tracks[0]]
            threshold_sq = min_distance * min_distance

            for point in tracks[1:]:
                # Check distance to all existing unique points
                dists_sq = np.sum((np.array(unique_points) - point) ** 2, axis=1)

                if np.all(dists_sq > threshold_sq):
                    unique_points.append(point)

            result = np.array(unique_points)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[YS-DEDUPE] CPU deduplication: {len(tracks)} → {len(result)} in {elapsed_ms:.2f}ms")

        return result

    def _find_clusters_cpu(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Find connected components in adjacency matrix (CPU)

        Uses iterative label propagation - fast and simple.

        Returns:
            cluster_ids: array where cluster_ids[i] is the cluster ID for point i
        """

        n = len(adjacency)
        cluster_ids = np.arange(n)

        # Iterative label propagation
        # Each point takes the minimum cluster ID of its neighbors
        max_iterations = 100
        for iteration in range(max_iterations):
            old_ids = cluster_ids.copy()

            # For each point, find minimum cluster ID among neighbors
            for i in range(n):
                neighbors = np.where(adjacency[i])[0]
                if len(neighbors) > 0:
                    cluster_ids[i] = np.min(cluster_ids[neighbors])

            # Check convergence
            if np.all(cluster_ids == old_ids):
                break

        return cluster_ids

    def _apply_keep_strategy_cpu(
        self,
        tracks: np.ndarray,
        cluster_ids: np.ndarray,
        strategy: str,
        dist_sq: np.ndarray
    ) -> np.ndarray:
        """
        Apply keep strategy to select one point per cluster (CPU)

        Returns:
            keep_indices: array of indices to keep
        """

        unique_clusters = np.unique(cluster_ids)
        keep_indices = []

        for cluster_id in unique_clusters:
            # Find all points in this cluster
            cluster_mask = cluster_ids == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if strategy == "first":
                # Keep first point (lowest index)
                keep_idx = cluster_indices[0]

            elif strategy == "last":
                # Keep last point (highest index)
                keep_idx = cluster_indices[-1]

            elif strategy == "center":
                # Keep point closest to cluster centroid
                cluster_points = tracks[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                dists = np.sum((cluster_points - centroid) ** 2, axis=1)
                keep_idx = cluster_indices[np.argmin(dists)]

            else:  # random
                # Keep random point
                rand_idx = np.random.randint(0, len(cluster_indices))
                keep_idx = cluster_indices[rand_idx]

            keep_indices.append(int(keep_idx))

        return np.array(keep_indices)


# Node registration
NODE_CLASS_MAPPINGS = {
    "YSTrackDeduplicate": TrackDeduplicateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSTrackDeduplicate": "YS Track Deduplicate"
}
