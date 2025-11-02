"""
Enhanced Track Detect Node for YS-vision-tools
Provides 7+ smart detection methods with GPU acceleration
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..utils import (
    GPUAccelerator,
    get_gpu_accelerator,
    is_gpu_available,
    ensure_numpy_hwc,
    rgb_to_grayscale
)

# Try importing optional dependencies
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class EnhancedTrackDetectNode:
    """Advanced tracking with 7+ detection strategies and GPU acceleration"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_method": ([
                    "gradient_magnitude",  # Sobel/Scharr based
                    "phase_congruency",    # Frequency domain
                    "structure_tensor",    # Corner quality
                    "optical_flow",        # Motion-based
                    "saliency_map",        # Visual attention
                    "object_detection",    # YOLO-based (requires ultralytics)
                    "hybrid_adaptive"      # Combines multiple methods
                ],),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "points_per_frame": ("INT", {"default": 200, "min": 1, "max": 2000, "step": 1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Preprocessing
                "gamma_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Gradient-based options
                "gradient_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Area-based filtering
                "min_area": ("INT", {"default": 5, "min": 1, "max": 500, "step": 1}),
                "max_area": ("INT", {"default": 500, "min": 10, "max": 5000, "step": 10}),

                # Object detection options (requires YOLO)
                "object_classes": ("STRING", {"default": "person,car,face"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Optical flow options
                "flow_threshold": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "previous_frame": ("IMAGE",),

                # Advanced options
                "use_kalman_filter": ("BOOLEAN", {"default": False}),
                "temporal_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TRACKS", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("tracks", "count", "avg_confidence", "debug_viz")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def __init__(self):
        self.gpu = get_gpu_accelerator()
        self.yolo_model = None
        self.kalman_filters = {}
        self.previous_tracks = None

    def execute(self, image, detection_method, sensitivity,
                points_per_frame, use_gpu, **kwargs):
        """Execute advanced detection"""

        # DEBUG
        print(f"\n[YS-TRACK] Input image type: {type(image)}, shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")

        # Check if batch input
        import torch
        is_batch = False
        batch_size = 1

        if torch.is_tensor(image):
            if len(image.shape) == 4 and image.shape[0] > 1:
                is_batch = True
                batch_size = image.shape[0]
                print(f"[YS-TRACK] BATCH MODE: Processing {batch_size} frames")

        if is_batch:
            # Process each frame in batch
            all_tracks = []
            for i in range(batch_size):
                frame = image[i:i+1]  # Keep batch dim
                frame_np = ensure_numpy_hwc(frame)
                tracks_single, _ = self._process_single_frame(frame_np, detection_method, sensitivity, points_per_frame, use_gpu, **kwargs)
                all_tracks.append(tracks_single)
                print(f"[YS-TRACK] Frame {i}: {len(tracks_single)} points")

            # Return list of track arrays
            print(f"[YS-TRACK] Returning list of {len(all_tracks)} track arrays")
            return (all_tracks, sum(len(t) for t in all_tracks), 0.0, image[0:1])  # Return first frame as debug viz

        # Single frame mode
        image_np = ensure_numpy_hwc(image)
        print(f"[YS-TRACK] SINGLE MODE: {image_np.shape}")

        tracks, features = self._process_single_frame(image_np, detection_method, sensitivity, points_per_frame, use_gpu, **kwargs)

        count = len(tracks)
        avg_confidence = float(np.mean(features)) if len(features) > 0 else 0.0
        debug_viz = self._create_debug_viz(image_np, tracks, features)

        return (tracks, count, avg_confidence, debug_viz)

    def _process_single_frame(self, image_np, detection_method, sensitivity, points_per_frame, use_gpu, **kwargs):
        """Process single frame - returns (tracks, features)"""

        # Gamma correction
        if kwargs.get('gamma_correction', 1.0) != 1.0:
            image_np = self._apply_gamma(image_np, kwargs['gamma_correction'], use_gpu and self.gpu.use_gpu)

        # Detection dispatch
        if detection_method == "gradient_magnitude":
            tracks, features = self._detect_gradient_based(image_np, sensitivity, use_gpu and self.gpu.use_gpu, **kwargs)
        elif detection_method == "phase_congruency":
            tracks, features = self._detect_phase_congruency(image_np, sensitivity, use_gpu and self.gpu.use_gpu)
        elif detection_method == "structure_tensor":
            tracks, features = self._detect_structure_tensor(image_np, sensitivity, use_gpu and self.gpu.use_gpu)
        elif detection_method == "optical_flow":
            tracks, features = self._detect_optical_flow(image_np, kwargs.get('previous_frame'), use_gpu and self.gpu.use_gpu, **kwargs)
        elif detection_method == "saliency_map":
            tracks, features = self._detect_saliency(image_np, sensitivity, use_gpu and self.gpu.use_gpu)
        elif detection_method == "object_detection":
            tracks, features = self._detect_objects(image_np, **kwargs)
        elif detection_method == "hybrid_adaptive":
            tracks, features = self._detect_hybrid(image_np, sensitivity, use_gpu and self.gpu.use_gpu, **kwargs)
        else:
            tracks, features = self._detect_gradient_based(image_np, sensitivity, use_gpu and self.gpu.use_gpu, **kwargs)

        # Filtering
        if len(tracks) > 0 and ('min_area' in kwargs or 'max_area' in kwargs):
            tracks, features = self._filter_by_area(tracks, features, **kwargs)

        if kwargs.get('use_kalman_filter', False) and len(tracks) > 0:
            tracks = self._apply_kalman_filter(tracks)

        # Limit points
        if len(tracks) > points_per_frame:
            indices = self._select_best_points(features, points_per_frame)
            tracks = tracks[indices]
            features = features[indices]

        return (tracks, features)

    def _apply_gamma(self, image: np.ndarray, gamma: float, use_gpu: bool) -> np.ndarray:
        """Apply gamma correction for better feature detection"""
        if use_gpu and CUPY_AVAILABLE:
            img_gpu = cp.asarray(image)
            corrected = cp.power(img_gpu, gamma)
            return cp.asnumpy(corrected)
        return np.power(image, gamma)

    def _detect_gradient_based(self, image: np.ndarray, sensitivity: float,
                               use_gpu: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced gradient-based detection using Sobel/Scharr"""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image

        if use_gpu and CUPY_AVAILABLE:
            img_gpu = cp.asarray(gray.astype(np.float32))

            # Scharr operator (more accurate than Sobel)
            grad_x = cp.asarray(cv2.Scharr(gray, cv2.CV_64F, 1, 0))
            grad_y = cp.asarray(cv2.Scharr(gray, cv2.CV_64F, 0, 1))

            # Gradient magnitude and angle
            magnitude = cp.sqrt(grad_x**2 + grad_y**2)
            angle = cp.arctan2(grad_y, grad_x)

            # Non-maximum suppression
            magnitude = self._gpu_non_max_suppression(magnitude, angle)

            # Threshold
            threshold = cp.percentile(magnitude, (1 - sensitivity) * 100)
            points = cp.argwhere(magnitude > threshold)

            # Convert back to CPU
            if len(points) > 0:
                tracks = cp.asnumpy(points[:, [1, 0]])  # Convert to (x, y)
                features = cp.asnumpy(magnitude[points[:, 0], points[:, 1]])
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)
        else:
            # CPU fallback
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            threshold = np.percentile(magnitude, (1 - sensitivity) * 100)

            points = np.argwhere(magnitude > threshold)
            if len(points) > 0:
                tracks = points[:, [1, 0]]  # Convert to (x, y)
                features = magnitude[points[:, 0], points[:, 1]]
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)

        return tracks, features

    def _detect_phase_congruency(self, image: np.ndarray, sensitivity: float,
                                 use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Detect features using phase congruency (frequency domain)"""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image

        if use_gpu and CUPY_AVAILABLE:
            img_gpu = cp.asarray(gray.astype(np.float32))

            # FFT-based phase congruency
            fft = cp.fft.fft2(img_gpu)

            # Multiple scale filters (simplified log-Gabor)
            scales = [4, 8, 16, 32]
            orientations = [0, 45, 90, 135]

            pc_sum = cp.zeros_like(img_gpu)

            for scale in scales:
                for orient in orientations:
                    # Create simple band-pass filter
                    filter_kernel = self._create_bandpass_filter_gpu(
                        img_gpu.shape, scale, orient
                    )

                    # Apply filter in frequency domain
                    filtered = cp.fft.ifft2(fft * filter_kernel)

                    # Accumulate magnitude
                    pc_sum += cp.abs(filtered)

            # Normalize
            pc_sum /= (len(scales) * len(orientations))

            # Find peaks
            threshold = cp.percentile(pc_sum, (1 - sensitivity) * 100)
            points = cp.argwhere(pc_sum > threshold)

            if len(points) > 0:
                tracks = cp.asnumpy(points[:, [1, 0]])
                features = cp.asnumpy(pc_sum[points[:, 0], points[:, 1]])
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)
        else:
            # Simplified CPU version using Canny edge detector
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            points = np.argwhere(edges > 0)

            if len(points) > 0:
                tracks = points[:, [1, 0]]
                features = np.ones(len(tracks))
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)

        return tracks, features

    def _detect_structure_tensor(self, image: np.ndarray, sensitivity: float,
                                 use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Detect using structure tensor (Harris corner variant)"""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image

        if use_gpu and CUPY_AVAILABLE:
            img_gpu = cp.asarray(gray.astype(np.float32))

            # Compute gradients
            Ix = cp.asarray(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
            Iy = cp.asarray(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

            # Structure tensor components
            Ixx = Ix * Ix
            Iyy = Iy * Iy
            Ixy = Ix * Iy

            # Gaussian smoothing
            Ixx_smooth = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Ixx), (5, 5), 1.5))
            Iyy_smooth = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Iyy), (5, 5), 1.5))
            Ixy_smooth = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Ixy), (5, 5), 1.5))

            # Compute eigenvalues (Shi-Tomasi corner response)
            trace = Ixx_smooth + Iyy_smooth
            det = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth

            # Corner response: minimum eigenvalue
            response = trace / 2 - cp.sqrt((trace / 2) ** 2 - det + 1e-6)

            # Find peaks
            threshold = cp.percentile(response[response > 0], (1 - sensitivity) * 100)
            points = cp.argwhere(response > threshold)

            if len(points) > 0:
                tracks = cp.asnumpy(points[:, [1, 0]])
                features = cp.asnumpy(response[points[:, 0], points[:, 1]])
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)
        else:
            # CPU version using OpenCV
            gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

            corners = cv2.goodFeaturesToTrack(
                gray_uint8,
                maxCorners=2000,
                qualityLevel=1 - sensitivity,
                minDistance=5
            )

            if corners is not None:
                tracks = corners.reshape(-1, 2)
                features = np.ones(len(tracks))
            else:
                tracks = np.empty((0, 2))
                features = np.empty(0)

        return tracks, features

    def _detect_optical_flow(self, image: np.ndarray, previous_frame: Optional[Any],
                            use_gpu: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Optical flow-based detection"""

        if previous_frame is None:
            # Fall back to gradient-based detection
            return self._detect_gradient_based(image, 0.5, use_gpu, **kwargs)

        # Convert both frames to grayscale
        if len(image.shape) == 3:
            current_gray = rgb_to_grayscale(image)
        else:
            current_gray = image

        previous_np = ensure_numpy_hwc(previous_frame)
        if len(previous_np.shape) == 3:
            previous_gray = rgb_to_grayscale(previous_np)
        else:
            previous_gray = previous_np

        # Convert to uint8
        current_gray_uint8 = (current_gray * 255).astype(np.uint8) if current_gray.max() <= 1.0 else current_gray.astype(np.uint8)
        previous_gray_uint8 = (previous_gray * 255).astype(np.uint8) if previous_gray.max() <= 1.0 else previous_gray.astype(np.uint8)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray_uint8, current_gray_uint8,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Compute flow magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold based on flow magnitude
        flow_threshold = kwargs.get('flow_threshold', 2.0)
        mask = magnitude > flow_threshold

        points = np.argwhere(mask)

        if len(points) > 0:
            tracks = points[:, [1, 0]]  # Convert to (x, y)
            features = magnitude[points[:, 0], points[:, 1]]
        else:
            tracks = np.empty((0, 2))
            features = np.empty(0)

        return tracks, features

    def _detect_saliency(self, image: np.ndarray, sensitivity: float,
                        use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Detect using saliency map (visual attention)"""

        # Convert to uint8 for OpenCV saliency
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        # Use OpenCV's static saliency detector
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(image_uint8)

        if not success:
            return np.empty((0, 2)), np.empty(0)

        # Threshold saliency map
        threshold = np.percentile(saliency_map, (1 - sensitivity) * 100)
        mask = saliency_map > threshold

        points = np.argwhere(mask)

        if len(points) > 0:
            tracks = points[:, [1, 0]]  # Convert to (x, y)
            features = saliency_map[points[:, 0], points[:, 1]]
        else:
            tracks = np.empty((0, 2))
            features = np.empty(0)

        return tracks, features

    def _detect_objects(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Object detection using YOLO"""

        if not YOLO_AVAILABLE:
            print("Warning: YOLO not available. Install ultralytics for object detection.")
            return self._detect_gradient_based(image, 0.5, False, **kwargs)

        if self.yolo_model is None:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed

        # Convert to uint8 for YOLO
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        # Run inference
        results = self.yolo_model(image_uint8, verbose=False)

        # Filter by confidence
        conf_threshold = kwargs.get('confidence_threshold', 0.5)

        tracks = []
        features = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if float(box.conf) > conf_threshold:
                        # Get box center as track point
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        tracks.append([center_x, center_y])
                        features.append(float(box.conf))

        if len(tracks) > 0:
            return np.array(tracks), np.array(features)
        else:
            return np.empty((0, 2)), np.empty(0)

    def _detect_hybrid(self, image: np.ndarray, sensitivity: float,
                      use_gpu: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid detection combining multiple methods"""

        # Combine gradient and structure tensor
        tracks1, features1 = self._detect_gradient_based(image, sensitivity, use_gpu, **kwargs)
        tracks2, features2 = self._detect_structure_tensor(image, sensitivity, use_gpu)

        if len(tracks1) == 0 and len(tracks2) == 0:
            return np.empty((0, 2)), np.empty(0)
        elif len(tracks1) == 0:
            return tracks2, features2
        elif len(tracks2) == 0:
            return tracks1, features1

        # Merge and deduplicate
        all_tracks = np.vstack([tracks1, tracks2])
        all_features = np.hstack([features1, features2])

        # Remove duplicates within radius
        unique_indices = self._remove_duplicates(all_tracks, radius=10)

        return all_tracks[unique_indices], all_features[unique_indices]

    def _gpu_non_max_suppression(self, magnitude: 'cp.ndarray',
                                 angle: 'cp.ndarray') -> 'cp.ndarray':
        """GPU-accelerated non-maximum suppression"""

        # Simplified NMS for GPU
        # Dilate and compare with original
        from cupyx.scipy import ndimage as cp_ndimage

        dilated = cp_ndimage.maximum_filter(magnitude, size=3)
        suppressed = cp.where(magnitude == dilated, magnitude, 0)

        return suppressed

    def _create_bandpass_filter_gpu(self, shape: Tuple[int, int],
                                    scale: int, orientation: float) -> 'cp.ndarray':
        """Create simple band-pass filter in frequency domain on GPU"""

        rows, cols = shape
        u = cp.fft.fftfreq(cols).reshape(1, -1)
        v = cp.fft.fftfreq(rows).reshape(-1, 1)

        radius = cp.sqrt(u**2 + v**2)

        # Simple band-pass
        center_freq = 1.0 / scale
        bandwidth = 0.5

        band_pass = cp.exp(-((radius - center_freq) ** 2) / (2 * bandwidth ** 2))
        band_pass[radius == 0] = 0

        return band_pass

    def _filter_by_area(self, tracks: np.ndarray, features: np.ndarray,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Filter points by area (placeholder for connected component analysis)"""
        # Simplified: just return all tracks for now
        # In a full implementation, this would analyze connected components
        return tracks, features

    def _apply_kalman_filter(self, tracks: np.ndarray) -> np.ndarray:
        """Apply Kalman filtering for temporal stability (simplified)"""
        # Simplified: weighted average with previous tracks
        if self.previous_tracks is not None and len(self.previous_tracks) == len(tracks):
            alpha = 0.7  # Smoothing factor
            tracks = alpha * tracks + (1 - alpha) * self.previous_tracks

        return tracks

    def _remove_duplicates(self, tracks: np.ndarray, radius: float) -> np.ndarray:
        """Remove duplicate points within radius"""
        if len(tracks) == 0:
            return np.array([])

        from scipy.spatial import cKDTree

        tree = cKDTree(tracks)
        unique_indices = []
        used = set()

        for i, point in enumerate(tracks):
            if i in used:
                continue

            unique_indices.append(i)

            # Find all points within radius
            neighbors = tree.query_ball_point(point, radius)
            used.update(neighbors)

        return np.array(unique_indices)

    def _select_best_points(self, features: np.ndarray, count: int) -> np.ndarray:
        """Select best points based on feature strength"""
        if len(features) <= count:
            return np.arange(len(features))

        # Sort by feature strength and take top count
        indices = np.argsort(features)[::-1][:count]
        return indices

    def _create_debug_viz(self, image: np.ndarray, tracks: np.ndarray,
                          features: np.ndarray) -> np.ndarray:
        """Create debug visualization"""

        # Create visualization
        if image.max() <= 1.0:
            viz = (image * 255).astype(np.uint8)
        else:
            viz = image.astype(np.uint8)

        # Ensure RGB
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)
        elif viz.shape[2] == 4:
            viz = viz[:, :, :3]

        # Draw points
        for i, (x, y) in enumerate(tracks.astype(int)):
            if 0 <= x < viz.shape[1] and 0 <= y < viz.shape[0]:
                # Color by feature strength
                strength = features[i] if i < len(features) else 1.0
                color_intensity = min(255, int(strength * 255))
                cv2.circle(viz, (int(x), int(y)), 3, (color_intensity, 255 - color_intensity, 0), -1)

        # Convert back to float [0, 1]
        viz_float = viz.astype(np.float32) / 255.0

        # Add batch dimension for ComfyUI
        from ..utils import numpy_to_comfyui
        return numpy_to_comfyui(viz_float)


# ComfyUI node registration information
NODE_CLASS_MAPPINGS = {
    "YS_TrackDetect": EnhancedTrackDetectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_TrackDetect": "Track Detect (Enhanced) 🎯"
}
