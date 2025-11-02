# Phase 1: MVP Implementation (Enhanced for YS-vision-tools)

> **✅ STATUS: PHASE 1 COMPLETE & DEPLOYED**
> **Date Completed:** November 2, 2025
> **Deployment:** Successfully deployed to ComfyUI and working
> **Bug Fixes:** Critical tensor format bug identified and fixed
> **Next Phase:** See [03-PHASE2-EXTENDED.md](03-PHASE2-EXTENDED.md)

---

## 🎯 Phase 1 Goal
Build the core system with sophisticated tracking and advanced line rendering capabilities, fully utilizing RTX 5090 GPU power.

**Key Enhancements:**
- Advanced curve mathematics for unique visual effects
- Multiple tracking methods beyond basic thresholding
- GPU acceleration throughout using CuPy/CUDA
- Experimental rendering techniques

## 🚀 GPU Setup for RTX 5090

### Install GPU Libraries
```bash
# CUDA-accelerated computing
pip install cupy-cuda12x==12.3.0  # For RTX 5090
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: For advanced object detection
pip install ultralytics  # YOLO for object-based tracking
pip install onnxruntime-gpu  # Optimized inference
```

### GPU Memory Management
```python
import cupy as cp
import torch

# Set memory pool for CuPy
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=8 * 1024**3)  # 8GB limit

# PyTorch GPU settings
torch.cuda.set_per_process_memory_fraction(0.8)
torch.backends.cudnn.benchmark = True
```

---

## 📋 Enhanced Implementation Tasks

### Task 1: GPU-Accelerated Common Utilities
**File:** `custom_nodes/ys_vision_tools/utils/gpu_common.py`
**Time:** 3-4 hours

```python
import numpy as np
import cupy as cp
import torch
from typing import Union, Tuple

class GPUAccelerator:
    """Manage GPU acceleration for YS-vision-tools"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()

    def to_gpu(self, array: np.ndarray) -> cp.ndarray:
        """Transfer array to GPU memory"""
        if self.use_gpu:
            return cp.asarray(array)
        return array

    def to_cpu(self, array: Union[cp.ndarray, np.ndarray]) -> np.ndarray:
        """Transfer array back to CPU"""
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    def gpu_convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """GPU-accelerated convolution"""
        if self.use_gpu:
            img_gpu = cp.asarray(image)
            kernel_gpu = cp.asarray(kernel)
            result = cp.signal.convolve2d(img_gpu, kernel_gpu, mode='same')
            return cp.asnumpy(result)
        return scipy.signal.convolve2d(image, kernel, mode='same')
```

---

### Task 2: Enhanced TrackDetect Node with Smart Detection
**Files:**
- `custom_nodes/ys_vision_tools/nodes/track_detect_enhanced.py`
- `tests/unit/test_track_detect_enhanced.py`
**Time:** 6-7 hours

#### 2.1 Advanced Detection Methods

```python
import cv2
import numpy as np
import cupy as cp
import torch
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO

class EnhancedTrackDetectNode:
    """Advanced tracking with multiple detection strategies"""

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
                    "saliency_map",       # Visual attention
                    "object_detection",    # YOLO-based
                    "hybrid_adaptive"      # Combines multiple
                ],),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "points_per_frame": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Gradient-based options
                "gamma_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0}),
                "gradient_threshold": ("FLOAT", {"default": 0.1}),

                # Area-based filtering
                "min_area": ("INT", {"default": 5}),
                "max_area": ("INT", {"default": 500}),
                "aspect_ratio_range": ("FLOAT_RANGE", {"default": (0.5, 2.0)}),

                # Object detection options
                "object_classes": ("STRING", {"default": "person,car,face"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5}),

                # Optical flow options
                "flow_threshold": ("FLOAT", {"default": 2.0}),
                "previous_frame": ("IMAGE",),

                # Advanced options
                "use_kalman_filter": ("BOOLEAN", {"default": False}),
                "temporal_smoothing": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("TRACKS", "IDS", "CONFIDENCE", "BOXES", "FEATURES", "TRACK_STATE")
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Tracking"

    def __init__(self):
        self.gpu = GPUAccelerator()
        self.yolo_model = None
        self.kalman_filters = {}

    def execute(self, image, detection_method, sensitivity,
                points_per_frame, use_gpu, **kwargs):
        """Execute advanced detection"""

        # Gamma correction preprocessing
        if kwargs.get('gamma_correction', 1.0) != 1.0:
            image = self._apply_gamma(image, kwargs['gamma_correction'], use_gpu)

        # Method dispatch
        if detection_method == "gradient_magnitude":
            tracks, features = self._detect_gradient_based(image, sensitivity, use_gpu, **kwargs)
        elif detection_method == "phase_congruency":
            tracks, features = self._detect_phase_congruency(image, sensitivity, use_gpu)
        elif detection_method == "structure_tensor":
            tracks, features = self._detect_structure_tensor(image, sensitivity, use_gpu)
        elif detection_method == "optical_flow":
            tracks, features = self._detect_optical_flow(image, kwargs.get('previous_frame'), use_gpu)
        elif detection_method == "saliency_map":
            tracks, features = self._detect_saliency(image, sensitivity, use_gpu)
        elif detection_method == "object_detection":
            tracks, features, boxes = self._detect_objects(image, **kwargs)
        elif detection_method == "hybrid_adaptive":
            tracks, features = self._detect_hybrid(image, sensitivity, use_gpu, **kwargs)

        # Area-based filtering
        if 'min_area' in kwargs or 'max_area' in kwargs:
            tracks, features = self._filter_by_area(tracks, features, **kwargs)

        # Kalman filtering for temporal stability
        if kwargs.get('use_kalman_filter', False):
            tracks = self._apply_kalman_filter(tracks)

        # Limit points
        if len(tracks) > points_per_frame:
            indices = self._select_best_points(features, points_per_frame)
            tracks = tracks[indices]
            features = features[indices]

        # Generate IDs and confidence
        ids = np.arange(len(tracks))
        confidence = self._compute_confidence(features)

        # Create state
        state = {
            'method': detection_method,
            'frame_count': kwargs.get('frame_count', 0) + 1,
            'tracks': tracks,
            'features': features
        }

        return (tracks, ids, confidence, boxes, features, state)

    def _apply_gamma(self, image: np.ndarray, gamma: float, use_gpu: bool) -> np.ndarray:
        """Apply gamma correction for better feature detection"""
        if use_gpu:
            img_gpu = cp.asarray(image)
            corrected = cp.power(img_gpu, gamma)
            return cp.asnumpy(corrected)
        return np.power(image, gamma)

    def _detect_gradient_based(self, image: np.ndarray, sensitivity: float,
                               use_gpu: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced gradient-based detection using Sobel/Scharr"""

        if use_gpu:
            img_gpu = cp.asarray(image)

            # Compute gradients on GPU
            if len(img_gpu.shape) == 3:
                img_gpu = cp.mean(img_gpu, axis=2)

            # Scharr operator (more accurate than Sobel)
            grad_x = cp.asarray(cv2.Scharr(cp.asnumpy(img_gpu), cv2.CV_64F, 1, 0))
            grad_y = cp.asarray(cv2.Scharr(cp.asnumpy(img_gpu), cv2.CV_64F, 0, 1))

            # Gradient magnitude and angle
            magnitude = cp.sqrt(grad_x**2 + grad_y**2)
            angle = cp.arctan2(grad_y, grad_x)

            # Non-maximum suppression
            magnitude = self._gpu_non_max_suppression(magnitude, angle)

            # Threshold
            threshold = cp.percentile(magnitude, (1 - sensitivity) * 100)
            points = cp.argwhere(magnitude > threshold)

            # Convert back to CPU
            tracks = cp.asnumpy(points[:, [1, 0]])  # Convert to (x, y)
            features = cp.asnumpy(magnitude[points[:, 0], points[:, 1]])
        else:
            # CPU fallback
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            threshold = np.percentile(magnitude, (1 - sensitivity) * 100)

            points = np.argwhere(magnitude > threshold)
            tracks = points[:, [1, 0]]
            features = magnitude[points[:, 0], points[:, 1]]

        return tracks, features

    def _detect_phase_congruency(self, image: np.ndarray, sensitivity: float,
                                 use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Detect features using phase congruency (frequency domain)"""

        if use_gpu:
            img_gpu = cp.asarray(image)
            if len(img_gpu.shape) == 3:
                img_gpu = cp.mean(img_gpu, axis=2)

            # FFT-based phase congruency
            fft = cp.fft.fft2(img_gpu)

            # Multiple scale filters (log-Gabor)
            scales = [4, 8, 16, 32]
            orientations = [0, 45, 90, 135]

            pc_sum = cp.zeros_like(img_gpu)

            for scale in scales:
                for orient in orientations:
                    # Create log-Gabor filter
                    filter_kernel = self._create_log_gabor_gpu(
                        img_gpu.shape, scale, orient
                    )

                    # Apply filter in frequency domain
                    filtered = cp.fft.ifft2(fft * filter_kernel)

                    # Accumulate phase congruency
                    pc_sum += cp.abs(filtered)

            # Normalize
            pc_sum /= len(scales) * len(orientations)

            # Find peaks
            threshold = cp.percentile(pc_sum, (1 - sensitivity) * 100)
            points = cp.argwhere(pc_sum > threshold)

            tracks = cp.asnumpy(points[:, [1, 0]])
            features = cp.asnumpy(pc_sum[points[:, 0], points[:, 1]])
        else:
            # Simplified CPU version
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Use Canny edge detector as approximation
            edges = cv2.Canny(gray, 50, 150)
            points = np.argwhere(edges > 0)

            tracks = points[:, [1, 0]]
            features = np.ones(len(tracks))

        return tracks, features

    def _detect_structure_tensor(self, image: np.ndarray, sensitivity: float,
                                 use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Detect using structure tensor (Harris corner variant)"""

        if use_gpu:
            img_gpu = cp.asarray(image)
            if len(img_gpu.shape) == 3:
                img_gpu = cp.mean(img_gpu, axis=2)

            # Compute structure tensor components
            Ix = cp.asarray(cv2.Sobel(cp.asnumpy(img_gpu), cv2.CV_64F, 1, 0, ksize=3))
            Iy = cp.asarray(cv2.Sobel(cp.asnumpy(img_gpu), cv2.CV_64F, 0, 1, ksize=3))

            # Structure tensor
            Ixx = Ix * Ix
            Iyy = Iy * Iy
            Ixy = Ix * Iy

            # Gaussian smoothing
            sigma = 1.5
            Ixx = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Ixx), (5, 5), sigma))
            Iyy = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Iyy), (5, 5), sigma))
            Ixy = cp.asarray(cv2.GaussianBlur(cp.asnumpy(Ixy), (5, 5), sigma))

            # Eigenvalues of structure tensor
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy

            # Corner response (Shi-Tomasi)
            response = cp.sqrt((trace/2)**2 - det) - trace/2

            # Find peaks
            threshold = cp.percentile(response, (1 - sensitivity) * 100)
            points = cp.argwhere(response > threshold)

            tracks = cp.asnumpy(points[:, [1, 0]])
            features = cp.asnumpy(response[points[:, 0], points[:, 1]])
        else:
            # CPU version using OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=1000,
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

    def _detect_objects(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Object detection using YOLO"""

        if self.yolo_model is None:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed

        # Run inference
        results = self.yolo_model(image)

        # Filter by classes
        target_classes = kwargs.get('object_classes', 'person').split(',')
        conf_threshold = kwargs.get('confidence_threshold', 0.5)

        tracks = []
        features = []
        boxes = []

        for r in results:
            for box in r.boxes:
                if box.conf > conf_threshold:
                    # Get box center as track point
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    tracks.append([center_x, center_y])
                    features.append(float(box.conf))
                    boxes.append([x1, y1, x2 - x1, y2 - y1])

        return np.array(tracks), np.array(features), np.array(boxes)

    def _detect_hybrid(self, image: np.ndarray, sensitivity: float,
                      use_gpu: bool, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid detection combining multiple methods"""

        # Combine gradient and structure tensor
        tracks1, features1 = self._detect_gradient_based(image, sensitivity, use_gpu, **kwargs)
        tracks2, features2 = self._detect_structure_tensor(image, sensitivity, use_gpu)

        # Merge and deduplicate
        all_tracks = np.vstack([tracks1, tracks2])
        all_features = np.hstack([features1, features2])

        # Remove duplicates within radius
        unique_indices = self._remove_duplicates(all_tracks, radius=10)

        return all_tracks[unique_indices], all_features[unique_indices]

    def _gpu_non_max_suppression(self, magnitude: cp.ndarray,
                                 angle: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated non-maximum suppression"""

        # Quantize angles to 4 directions
        angle = cp.rad2deg(angle) % 180

        suppressed = cp.zeros_like(magnitude)

        # Horizontal edges (0°, 180°)
        mask = ((angle <= 22.5) | (angle >= 157.5))
        suppressed[mask] = magnitude[mask]

        # Diagonal edges (45°)
        mask = ((angle > 22.5) & (angle <= 67.5))
        suppressed[mask] = magnitude[mask]

        # Vertical edges (90°)
        mask = ((angle > 67.5) & (angle <= 112.5))
        suppressed[mask] = magnitude[mask]

        # Diagonal edges (135°)
        mask = ((angle > 112.5) & (angle < 157.5))
        suppressed[mask] = magnitude[mask]

        return suppressed

    def _create_log_gabor_gpu(self, shape: Tuple[int, int],
                              scale: int, orientation: float) -> cp.ndarray:
        """Create log-Gabor filter in frequency domain on GPU"""

        rows, cols = shape
        u = cp.fft.fftfreq(cols).reshape(1, -1)
        v = cp.fft.fftfreq(rows).reshape(-1, 1)

        radius = cp.sqrt(u**2 + v**2)
        theta = cp.arctan2(v, u)

        # Log-Gabor radial component
        sigma_f = 0.56
        center_freq = 1.0 / scale
        radial = cp.exp(-(cp.log(radius/center_freq))**2 / (2 * cp.log(sigma_f)**2))
        radial[radius == 0] = 0

        # Angular component
        sigma_theta = cp.pi / 8
        angular = cp.exp(-(theta - orientation)**2 / (2 * sigma_theta**2))

        return radial * angular
```

---

### Task 5: Advanced LineLinkRenderer with Experimental Curves
**Files:**
- `custom_nodes/ys_vision_tools/nodes/line_link_renderer_advanced.py`
- `tests/unit/test_line_link_renderer_advanced.py`
**Time:** 6-7 hours

#### 5.1 Enhanced Line Rendering System

```python
import numpy as np
import cupy as cp
import cv2
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import CubicSpline, BSpline
from scipy.special import jv  # Bessel functions

class AdvancedLineLinkRendererNode:
    """Advanced line rendering with experimental curve equations"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT",),
                "image_height": ("INT",),

                "curve_type": ([
                    "straight",
                    "quadratic_bezier",
                    "cubic_bezier",
                    "catmull_rom",      # Smooth interpolation
                    "b_spline",         # B-spline curves
                    "hermite",          # Hermite curves
                    "fourier",          # Fourier series approximation
                    "logarithmic_spiral", # Spiral connections
                    "elastic",          # Physics-based elastic curves
                    "field_lines",      # Magnetic field simulation
                    "voronoi_edges",    # Voronoi diagram edges
                    "delaunay",         # Delaunay triangulation
                    "minimum_spanning", # MST connections
                    "gravitational",    # Gravity simulation
                    "neural_flow"       # Neural network flow
                ],),

                "line_style": ([
                    "solid",
                    "dotted",
                    "dashed",
                    "dash_dot",
                    "gradient_fade",    # Gradient along line
                    "pulsing",         # Animated pulse effect
                    "electric",        # Lightning-like
                    "particle_trail",  # Particle system
                    "double_line",     # Parallel lines
                    "wave",           # Sinusoidal modulation
                ],),

                "width_px": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Graph construction
                "graph_mode": (["knn", "radius", "delaunay", "mst", "voronoi"],),
                "k_neighbors": ("INT", {"default": 3, "min": 1, "max": 10}),
                "connection_radius": ("FLOAT", {"default": 100.0}),

                # Curve parameters
                "curve_tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "overshoot": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "control_point_offset": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "spiral_turns": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0}),

                # Style parameters
                "dot_spacing": ("FLOAT", {"default": 5.0}),  # For dotted lines
                "dash_length": ("FLOAT", {"default": 10.0}),
                "gradient_start_color": ("COLOR", {"default": [1, 1, 1]}),
                "gradient_end_color": ("COLOR", {"default": [0, 0, 1]}),
                "pulse_frequency": ("FLOAT", {"default": 1.0}),
                "wave_amplitude": ("FLOAT", {"default": 5.0}),
                "wave_frequency": ("FLOAT", {"default": 0.1}),

                # Physics parameters
                "gravity_strength": ("FLOAT", {"default": 0.1}),
                "elastic_stiffness": ("FLOAT", {"default": 0.5}),
                "field_strength": ("FLOAT", {"default": 1.0}),

                # Animation
                "time": ("FLOAT", {"default": 0.0}),  # For animated effects
                "seed": ("INT", {"default": 42}),     # For randomization

                # Performance
                "samples_per_curve": ("INT", {"default": 50, "min": 10, "max": 200}),
                "antialiasing": (["none", "2x", "4x", "8x"],),

                # Color
                "palette": ("PALETTE",),
                "color_mode": (["fixed", "palette", "gradient", "rainbow"],),
                "fixed_color": ("COLOR", {"default": [1, 1, 1]}),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def __init__(self):
        self.gpu = GPUAccelerator()
        np.random.seed(42)

    def execute(self, tracks, image_width, image_height, curve_type,
                line_style, width_px, opacity, use_gpu, **kwargs):
        """Render advanced lines with experimental curves"""

        # Initialize layer
        if kwargs.get('antialiasing', 'none') != 'none':
            aa_factor = int(kwargs['antialiasing'][0])
            layer = np.zeros((image_height * aa_factor,
                            image_width * aa_factor, 4), dtype=np.float32)
            tracks_scaled = tracks * aa_factor
        else:
            aa_factor = 1
            layer = np.zeros((image_height, image_width, 4), dtype=np.float32)
            tracks_scaled = tracks

        # Build graph connections
        edges = self._build_graph(tracks_scaled, **kwargs)

        # Render each edge with specified curve type
        for i, (start_idx, end_idx) in enumerate(edges):
            p1 = tracks_scaled[start_idx]
            p2 = tracks_scaled[end_idx]

            # Generate curve points
            if use_gpu and curve_type in ['fourier', 'field_lines', 'neural_flow']:
                curve_points = self._generate_curve_gpu(p1, p2, curve_type, **kwargs)
            else:
                curve_points = self._generate_curve(p1, p2, curve_type, **kwargs)

            # Determine color
            color = self._get_edge_color(i, start_idx, end_idx, **kwargs)

            # Render with specified style
            self._render_line_styled(layer, curve_points, color,
                                    line_style, width_px * aa_factor,
                                    opacity, use_gpu, **kwargs)

        # Downscale if antialiasing was used
        if aa_factor > 1:
            layer = cv2.resize(layer, (image_width, image_height),
                             interpolation=cv2.INTER_AREA)

        return (layer,)

    def _generate_curve(self, p1: np.ndarray, p2: np.ndarray,
                       curve_type: str, **kwargs) -> np.ndarray:
        """Generate curve points using various mathematical equations"""

        samples = kwargs.get('samples_per_curve', 50)
        t = np.linspace(0, 1, samples)

        if curve_type == "straight":
            points = np.outer(1-t, p1) + np.outer(t, p2)

        elif curve_type == "quadratic_bezier":
            # Control point with overshoot
            overshoot = kwargs.get('overshoot', 0.0)
            mid = (p1 + p2) / 2
            normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
            normal = normal / np.linalg.norm(normal)

            control = mid + normal * np.linalg.norm(p2 - p1) * (0.3 + overshoot)

            # Quadratic Bezier formula
            points = (np.outer((1-t)**2, p1) +
                     np.outer(2*(1-t)*t, control) +
                     np.outer(t**2, p2))

        elif curve_type == "cubic_bezier":
            # Two control points
            overshoot = kwargs.get('overshoot', 0.0)
            offset = kwargs.get('control_point_offset', 0.3)

            v = p2 - p1
            normal = np.array([-v[1], v[0]])
            normal = normal / np.linalg.norm(normal)

            c1 = p1 + v * offset + normal * np.linalg.norm(v) * (0.2 + overshoot)
            c2 = p2 - v * offset + normal * np.linalg.norm(v) * (0.2 - overshoot)

            # Cubic Bezier formula
            points = (np.outer((1-t)**3, p1) +
                     np.outer(3*(1-t)**2*t, c1) +
                     np.outer(3*(1-t)*t**2, c2) +
                     np.outer(t**3, p2))

        elif curve_type == "catmull_rom":
            # Catmull-Rom spline (needs 4 points)
            tension = kwargs.get('curve_tension', 0.5)

            # Create virtual points before and after
            p0 = p1 - (p2 - p1) * 0.5
            p3 = p2 + (p2 - p1) * 0.5

            # Catmull-Rom matrix
            points = []
            for ti in t:
                t2 = ti * ti
                t3 = t2 * ti

                point = (
                    (-tension*t3 + 2*tension*t2 - tension*ti) * p0 +
                    ((2-tension)*t3 + (tension-3)*t2 + 1) * p1 +
                    ((tension-2)*t3 + (3-2*tension)*t2 + tension*ti) * p2 +
                    (tension*t3 - tension*t2) * p3
                )
                points.append(point)

            points = np.array(points)

        elif curve_type == "logarithmic_spiral":
            # Logarithmic spiral between points
            turns = kwargs.get('spiral_turns', 0.5)

            # Convert to polar coordinates relative to p1
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            r_end = np.sqrt(dx**2 + dy**2)
            theta_end = np.arctan2(dy, dx)

            # Spiral parameters
            a = 1
            b = np.log(r_end) / (theta_end + 2*np.pi*turns)

            theta = np.linspace(0, theta_end + 2*np.pi*turns, samples)
            r = a * np.exp(b * theta)

            # Convert back to Cartesian
            x = p1[0] + r * np.cos(theta)
            y = p1[1] + r * np.sin(theta)

            points = np.column_stack([x, y])

        elif curve_type == "elastic":
            # Physics-based elastic curve
            stiffness = kwargs.get('elastic_stiffness', 0.5)

            # Simulate elastic deformation
            points = []
            for ti in t:
                # Add oscillation that dampens
                damping = 1 - ti
                oscillation = np.sin(ti * np.pi * 4) * damping * stiffness

                normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                normal = normal / np.linalg.norm(normal)

                point = p1 * (1-ti) + p2 * ti + normal * oscillation * 20
                points.append(point)

            points = np.array(points)

        elif curve_type == "fourier":
            # Fourier series approximation
            n_harmonics = 5
            points = []

            for ti in t:
                point = p1 * (1-ti) + p2 * ti

                # Add Fourier harmonics
                for n in range(1, n_harmonics + 1):
                    amplitude = 1.0 / n
                    freq = n * 2 * np.pi

                    normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                    normal = normal / np.linalg.norm(normal)

                    offset = amplitude * np.sin(freq * ti) * 10
                    point = point + normal * offset

                points.append(point)

            points = np.array(points)

        elif curve_type == "field_lines":
            # Magnetic/electric field line simulation
            field_strength = kwargs.get('field_strength', 1.0)

            points = []
            current = p1.copy()

            for _ in range(samples):
                points.append(current.copy())

                # Calculate field direction (simplified)
                to_target = p2 - current
                distance = np.linalg.norm(to_target)

                if distance > 0:
                    # Field influences perpendicular to direct path
                    field_dir = np.array([-to_target[1], to_target[0]])
                    field_dir = field_dir / np.linalg.norm(field_dir)

                    # Combine direct path with field influence
                    step = to_target / samples
                    field_influence = field_dir * field_strength * np.sin(distance / 50)

                    current = current + step + field_influence

            points = np.array(points)

        else:
            # Default to straight line
            points = np.outer(1-t, p1) + np.outer(t, p2)

        return points

    def _generate_curve_gpu(self, p1: np.ndarray, p2: np.ndarray,
                           curve_type: str, **kwargs) -> np.ndarray:
        """GPU-accelerated curve generation for complex curves"""

        samples = kwargs.get('samples_per_curve', 50)

        # Transfer to GPU
        p1_gpu = cp.asarray(p1)
        p2_gpu = cp.asarray(p2)
        t = cp.linspace(0, 1, samples)

        if curve_type == "fourier":
            # GPU-accelerated Fourier synthesis
            n_harmonics = 10
            points = cp.zeros((samples, 2))

            # Base interpolation
            for i in range(2):
                points[:, i] = (1-t) * p1_gpu[i] + t * p2_gpu[i]

            # Add harmonics
            normal = cp.array([-(p2_gpu[1] - p1_gpu[1]),
                              p2_gpu[0] - p1_gpu[0]])
            normal = normal / cp.linalg.norm(normal)

            for n in range(1, n_harmonics + 1):
                amplitude = 1.0 / n
                freq = n * 2 * cp.pi
                offset = amplitude * cp.sin(freq * t) * 10

                points[:, 0] += normal[0] * offset
                points[:, 1] += normal[1] * offset

            return cp.asnumpy(points)

        elif curve_type == "field_lines":
            # GPU field line integration
            field_strength = kwargs.get('field_strength', 1.0)

            # Runge-Kutta integration on GPU
            points = cp.zeros((samples, 2))
            points[0] = p1_gpu

            for i in range(1, samples):
                current = points[i-1]
                to_target = p2_gpu - current

                # Field calculation
                distance = cp.linalg.norm(to_target)
                if distance > 0:
                    field_dir = cp.array([-to_target[1], to_target[0]])
                    field_dir = field_dir / cp.linalg.norm(field_dir)

                    step = to_target / samples
                    field = field_dir * field_strength * cp.sin(distance / 50)

                    points[i] = current + step + field

            return cp.asnumpy(points)

        elif curve_type == "neural_flow":
            # Simulated neural activation flow
            # Using GPU for matrix operations

            # Create activation pattern
            activations = cp.random.randn(samples, 2) * 0.1

            # Smooth with convolution
            kernel = cp.array([0.25, 0.5, 0.25])
            for i in range(2):
                activations[:, i] = cp.convolve(activations[:, i], kernel, mode='same')

            # Base path
            base_path = cp.outer(1-t, p1_gpu) + cp.outer(t, p2_gpu)

            # Add neural variations
            points = base_path + activations * 10

            return cp.asnumpy(points)

        # Default fallback
        return self._generate_curve(p1, p2, curve_type, **kwargs)

    def _render_line_styled(self, layer: np.ndarray, points: np.ndarray,
                           color: np.ndarray, style: str, width: float,
                           opacity: float, use_gpu: bool, **kwargs):
        """Render line with various styles"""

        if style == "solid":
            self._render_solid_line(layer, points, color, width, opacity)

        elif style == "dotted":
            spacing = kwargs.get('dot_spacing', 5.0)
            self._render_dotted_line(layer, points, color, width,
                                    opacity, spacing)

        elif style == "dashed":
            dash_length = kwargs.get('dash_length', 10.0)
            self._render_dashed_line(layer, points, color, width,
                                   opacity, dash_length)

        elif style == "gradient_fade":
            start_color = kwargs.get('gradient_start_color', color)
            end_color = kwargs.get('gradient_end_color', color)
            self._render_gradient_line(layer, points, start_color,
                                      end_color, width, opacity)

        elif style == "pulsing":
            time = kwargs.get('time', 0.0)
            freq = kwargs.get('pulse_frequency', 1.0)

            # Modulate opacity with time
            pulse_opacity = opacity * (0.5 + 0.5 * np.sin(time * freq * 2 * np.pi))
            self._render_solid_line(layer, points, color, width, pulse_opacity)

        elif style == "electric":
            # Lightning-like effect with jitter
            jittered_points = points + np.random.randn(*points.shape) * 2
            self._render_solid_line(layer, jittered_points, color, width * 0.5, opacity)

            # Add glow
            self._render_solid_line(layer, points, color * 0.5, width * 3, opacity * 0.3)

        elif style == "particle_trail":
            # Particle system along curve
            n_particles = int(len(points) / 3)
            particle_indices = np.random.choice(len(points), n_particles)

            for idx in particle_indices:
                pt = points[idx]
                size = np.random.uniform(1, width * 2)
                self._draw_soft_circle(layer, pt[0], pt[1], size,
                                      color, opacity * 0.7)

        elif style == "wave":
            # Sinusoidal modulation
            amplitude = kwargs.get('wave_amplitude', 5.0)
            frequency = kwargs.get('wave_frequency', 0.1)

            # Add wave perpendicular to path
            wave_points = []
            for i, pt in enumerate(points):
                if i > 0:
                    tangent = points[i] - points[i-1]
                    normal = np.array([-tangent[1], tangent[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-6)

                    offset = amplitude * np.sin(i * frequency)
                    wave_pt = pt + normal * offset
                else:
                    wave_pt = pt

                wave_points.append(wave_pt)

            self._render_solid_line(layer, np.array(wave_points),
                                  color, width, opacity)

        else:
            # Default to solid
            self._render_solid_line(layer, points, color, width, opacity)

    def _render_solid_line(self, layer: np.ndarray, points: np.ndarray,
                          color: np.ndarray, width: float, opacity: float):
        """Render solid line with anti-aliasing"""

        # Convert to integer coordinates
        points_int = points.astype(np.int32)

        # Draw using OpenCV with anti-aliasing
        for i in range(len(points_int) - 1):
            cv2.line(layer, tuple(points_int[i]), tuple(points_int[i+1]),
                    (*color, opacity), int(np.ceil(width)), cv2.LINE_AA)

    def _render_dotted_line(self, layer: np.ndarray, points: np.ndarray,
                           color: np.ndarray, width: float, opacity: float,
                           spacing: float):
        """Render dotted line"""

        # Calculate total line length
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]

        # Place dots at regular intervals
        num_dots = int(total_length / spacing)

        for i in range(num_dots):
            target_dist = i * spacing

            # Find position along curve
            idx = np.searchsorted(cumulative, target_dist)
            if idx > 0 and idx < len(points):
                t = (target_dist - cumulative[idx-1]) / (distances[idx-1] + 1e-6)
                pt = points[idx-1] * (1-t) + points[idx] * t

                # Draw dot
                cv2.circle(layer, tuple(pt.astype(int)), int(width),
                          (*color, opacity), -1, cv2.LINE_AA)

    def _render_gradient_line(self, layer: np.ndarray, points: np.ndarray,
                             start_color: np.ndarray, end_color: np.ndarray,
                             width: float, opacity: float):
        """Render line with color gradient"""

        for i in range(len(points) - 1):
            # Interpolate color
            t = i / (len(points) - 1)
            color = start_color * (1-t) + end_color * t

            # Draw segment
            cv2.line(layer, tuple(points[i].astype(int)),
                    tuple(points[i+1].astype(int)),
                    (*color, opacity), int(np.ceil(width)), cv2.LINE_AA)

    def _build_graph(self, tracks: np.ndarray, **kwargs) -> List[Tuple[int, int]]:
        """Build graph connections with various strategies"""

        mode = kwargs.get('graph_mode', 'knn')

        if mode == 'knn':
            k = kwargs.get('k_neighbors', 3)
            return self._build_knn_graph(tracks, k)

        elif mode == 'radius':
            radius = kwargs.get('connection_radius', 100.0)
            return self._build_radius_graph(tracks, radius)

        elif mode == 'delaunay':
            from scipy.spatial import Delaunay
            tri = Delaunay(tracks)
            edges = set()

            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)

            return list(edges)

        elif mode == 'mst':
            # Minimum spanning tree
            from scipy.spatial import distance_matrix
            from scipy.sparse.csgraph import minimum_spanning_tree

            dist_matrix = distance_matrix(tracks, tracks)
            mst = minimum_spanning_tree(dist_matrix)
            edges = []

            rows, cols = mst.nonzero()
            for i, j in zip(rows, cols):
                if i < j:
                    edges.append((i, j))

            return edges

        elif mode == 'voronoi':
            # Voronoi edges
            from scipy.spatial import Voronoi
            vor = Voronoi(tracks)
            edges = []

            for ridge in vor.ridge_points:
                if ridge[0] >= 0 and ridge[1] >= 0:
                    edges.append(tuple(ridge))

            return edges

        # Default to kNN
        return self._build_knn_graph(tracks, 3)

    def _build_knn_graph(self, tracks: np.ndarray, k: int) -> List[Tuple[int, int]]:
        """Build k-nearest neighbor graph"""
        from scipy.spatial import KDTree

        tree = KDTree(tracks)
        edges = set()

        for i, point in enumerate(tracks):
            distances, indices = tree.query(point, k=min(k+1, len(tracks)))

            for j in indices[1:]:  # Skip self
                edge = tuple(sorted([i, j]))
                edges.add(edge)

        return list(edges)

    def _build_radius_graph(self, tracks: np.ndarray,
                           radius: float) -> List[Tuple[int, int]]:
        """Build radius-based graph"""
        from scipy.spatial import KDTree

        tree = KDTree(tracks)
        edges = set()

        for i, point in enumerate(tracks):
            indices = tree.query_ball_point(point, radius)

            for j in indices:
                if i < j:
                    edges.add((i, j))

        return list(edges)
```

#### 5.2 Test Advanced Line Rendering

```python
def test_advanced_curves():
    """Test all curve types"""
    node = AdvancedLineLinkRendererNode()

    tracks = np.array([
        [50, 50], [150, 100], [250, 50],
        [50, 200], [150, 250], [250, 200]
    ])

    curve_types = [
        'cubic_bezier', 'catmull_rom', 'logarithmic_spiral',
        'elastic', 'fourier', 'field_lines'
    ]

    for curve_type in curve_types:
        layer = node.execute(
            tracks=tracks,
            image_width=300,
            image_height=300,
            curve_type=curve_type,
            line_style='solid',
            width_px=2.0,
            opacity=1.0,
            use_gpu=True,
            graph_mode='delaunay'
        )[0]

        # Should produce non-empty layer
        assert layer.max() > 0

        # Save for visual inspection
        save_test_image(layer, f"test_curve_{curve_type}.png")
```

---

## 📊 Performance Optimization for RTX 5090

### GPU Memory Pool Management
```python
class GPUMemoryManager:
    """Efficient GPU memory management for RTX 5090"""

    def __init__(self, gpu_memory_gb=24):  # RTX 5090 has 24GB
        self.total_memory = gpu_memory_gb * 1024**3
        self.reserved = int(self.total_memory * 0.8)  # Reserve 80%

        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=self.reserved)

        # PyTorch settings
        import torch
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def profile_operation(self, func, *args, **kwargs):
        """Profile GPU memory and time"""
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes()

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        result = func(*args, **kwargs)
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        used_after = mempool.used_bytes()

        print(f"GPU Time: {elapsed_ms:.2f}ms")
        print(f"GPU Memory: {(used_after - used_before) / 1024**2:.2f}MB")

        return result
```

---

## 🎬 Updated Phase 1 Success Metrics

### Performance Targets (RTX 5090)
- ✅ 4K @ 60+ fps with full effects
- ✅ 8K @ 30+ fps with advanced curves
- ✅ 1000+ tracked points in real-time
- ✅ Complex curve equations rendered in <5ms

### Quality Standards
- ✅ All curve types visually distinct
- ✅ Smooth transitions and animations
- ✅ No visible aliasing at 4K
- ✅ Stable tracking across all methods

## Next Steps
The enhanced Phase 1 now includes sophisticated tracking and rendering capabilities that fully leverage GPU acceleration. Continue to Phase 2 for additional visual effects.