"""
Track Merge Node for YS-vision-tools
Combines multiple track sources into a single unified output

Features:
- Merge 2+ track inputs into single output
- Handles both single-frame and batch (video) modes
- Compatible with all track-consuming nodes (line/dot renderers)
- Optional deduplication to remove overlapping points
"""

import numpy as np
from typing import Dict, Any, List, Optional


class TrackMergeNode:
    """
    Merge multiple track sources into a single unified track output

    Use Cases:
    - Combine detections from multiple methods
    - Mix manual + detected tracks
    - Layer effects from different sources
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks_a": ("TRACKS",),
                "tracks_b": ("TRACKS",),
            },
            "optional": {
                "tracks_c": ("TRACKS",),
                "tracks_d": ("TRACKS",),
                "deduplicate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove duplicate points within threshold distance"
                }),
                "dedup_threshold": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Distance threshold for duplicate detection (pixels)"
                }),
            }
        }

    RETURN_TYPES = ("TRACKS", "INT")
    RETURN_NAMES = ("tracks", "total_count")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def execute(self, tracks_a, tracks_b, tracks_c=None, tracks_d=None,
                deduplicate=False, dedup_threshold=2.0):
        """Merge multiple track sources"""

        print(f"\n[YS-MERGE] Executing Track Merge")
        print(f"[YS-MERGE] Deduplicate: {deduplicate}, Threshold: {dedup_threshold}")

        # Collect all non-None track inputs
        track_inputs = [tracks_a, tracks_b]
        if tracks_c is not None:
            track_inputs.append(tracks_c)
        if tracks_d is not None:
            track_inputs.append(tracks_d)

        print(f"[YS-MERGE] Merging {len(track_inputs)} track sources")

        # Check if any input is batch mode (list of arrays)
        is_batch = any(isinstance(t, list) for t in track_inputs)

        if is_batch:
            # BATCH MODE: Merge frame-by-frame
            merged_tracks, total = self._merge_batch(
                track_inputs, deduplicate, dedup_threshold
            )
        else:
            # SINGLE FRAME MODE: Merge directly
            merged_tracks, total = self._merge_single(
                track_inputs, deduplicate, dedup_threshold
            )

        print(f"[YS-MERGE] Output: {total} total points")
        return (merged_tracks, total)

    def _merge_batch(self, track_inputs: List, deduplicate: bool,
                     dedup_threshold: float):
        """Merge batch/video tracks frame-by-frame"""

        # Convert all inputs to list format
        normalized_inputs = []
        for tracks in track_inputs:
            if isinstance(tracks, list):
                normalized_inputs.append(tracks)
            else:
                # Single frame track -> wrap in list
                normalized_inputs.append([tracks])

        # Determine batch size (max length)
        batch_size = max(len(t) for t in normalized_inputs)
        print(f"[YS-MERGE] BATCH MODE: {batch_size} frames")

        merged_batch = []
        total_points = 0

        for i in range(batch_size):
            # Collect tracks for this frame
            frame_tracks = []
            for tracks_list in normalized_inputs:
                # Handle mismatched batch sizes (use last frame if shorter)
                frame_idx = min(i, len(tracks_list) - 1)
                frame_tracks.append(tracks_list[frame_idx])

            # Merge this frame
            merged_frame, frame_count = self._merge_single(
                frame_tracks, deduplicate, dedup_threshold
            )
            merged_batch.append(merged_frame)
            total_points += frame_count

            # Progress logging
            if i % 10 == 0 or i == batch_size - 1:
                print(f"[YS-MERGE] Processed frame {i+1}/{batch_size} ({frame_count} points)")

        avg_points = total_points // batch_size if batch_size > 0 else 0
        print(f"[YS-MERGE] Returning batch: {len(merged_batch)} frames, avg {avg_points} points/frame")

        return merged_batch, avg_points

    def _merge_single(self, track_inputs: List, deduplicate: bool,
                      dedup_threshold: float):
        """Merge single frame tracks"""

        # Convert all inputs to numpy arrays
        track_arrays = []
        for tracks in track_inputs:
            if isinstance(tracks, np.ndarray):
                if len(tracks) > 0:
                    track_arrays.append(tracks)
            elif isinstance(tracks, list) and len(tracks) > 0:
                track_arrays.append(np.array(tracks))

        if len(track_arrays) == 0:
            # No valid tracks
            return np.empty((0, 2)), 0

        # Concatenate all tracks
        merged = np.vstack(track_arrays)
        count = len(merged)

        # Optional deduplication
        if deduplicate and count > 1:
            merged = self._remove_duplicates(merged, dedup_threshold)
            deduped_count = len(merged)
            if deduped_count < count:
                print(f"[YS-MERGE] Deduplicated: {count} → {deduped_count} points")
            count = deduped_count

        return merged, count

    def _remove_duplicates(self, tracks: np.ndarray, threshold: float) -> np.ndarray:
        """
        Remove duplicate points within threshold distance.
        Uses simple pairwise distance check (CPU only for now).
        """

        if len(tracks) == 0:
            return tracks

        # Build list of unique points
        unique_points = [tracks[0]]  # Always keep first point
        threshold_sq = threshold * threshold

        for point in tracks[1:]:
            # Check distance to all existing unique points
            dists_sq = np.sum((np.array(unique_points) - point) ** 2, axis=1)

            # If not too close to any existing point, keep it
            if np.all(dists_sq > threshold_sq):
                unique_points.append(point)

        return np.array(unique_points)


# Node registration
NODE_CLASS_MAPPINGS = {
    "YSTrackMerge": TrackMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSTrackMerge": "YS Track Merge"
}
