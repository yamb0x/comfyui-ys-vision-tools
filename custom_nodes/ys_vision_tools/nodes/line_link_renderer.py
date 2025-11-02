"""
Advanced Line Link Renderer Node for YS-vision-tools
Renders sophisticated curves with 15+ mathematical types and 10+ animated styles
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

from ..utils import (
    get_gpu_accelerator,
    is_gpu_available,
    CurveGenerator,
    GraphBuilder,
    create_rgba_layer,
    numpy_to_comfyui
)

# Try importing optional dependencies
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class AdvancedLineLinkRendererNode:
    """Advanced line rendering with experimental curve equations and styles"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),

                "curve_type": ([
                    "straight",
                    "quadratic_bezier",
                    "cubic_bezier",
                    "catmull_rom",      # Smooth interpolation
                    "logarithmic_spiral", # Spiral connections
                    "elastic",          # Physics-based elastic curves
                    "fourier_series",   # Fourier series approximation
                    "field_lines",      # Magnetic field simulation
                    "gravitational",    # Gravity simulation
                    "delaunay",         # Delaunay triangulation edges
                    "voronoi_edges",    # Voronoi diagram edges
                    "minimum_spanning", # MST connections
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
                    "wave",           # Sinusoidal modulation
                ],),

                "width_px": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 20.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Graph construction
                "graph_mode": (["knn", "radius", "delaunay", "mst", "voronoi"],),
                "k_neighbors": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "connection_radius": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 10.0}),

                # Curve parameters
                "curve_tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "overshoot": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "control_point_offset": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "spiral_turns": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),

                # Style parameters
                "dot_spacing": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "dash_length": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 50.0, "step": 1.0}),
                "gradient_start_color": ("STRING", {"default": "1.0,1.0,1.0"}),  # RGB as string
                "gradient_end_color": ("STRING", {"default": "0.0,0.5,1.0"}),
                "pulse_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "wave_amplitude": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "wave_frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),

                # Physics parameters
                "gravity_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "elastic_stiffness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "field_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),

                # Animation
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "step": 1}),

                # Performance
                "samples_per_curve": ("INT", {"default": 50, "min": 10, "max": 200, "step": 10}),
                "antialiasing": (["none", "2x", "4x"],),

                # Color
                "fixed_color": ("STRING", {"default": "1.0,1.0,1.0"}),  # RGB as string
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def __init__(self):
        self.gpu = get_gpu_accelerator()
        np.random.seed(42)

    def execute(self, tracks, image_width, image_height, curve_type,
                line_style, width_px, opacity, use_gpu, **kwargs):
        """
        Render advanced lines with experimental curves.

        Args:
            preset: Preset configuration to use. If not "custom", applies preset settings
                    before using other parameters (user can still override)
        """

        # DEBUG
        print(f"\n[YS-LINE] Executing LineLinkRenderer")
        print(f"[YS-LINE] tracks type: {type(tracks)}")
        
        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            print(f"[YS-LINE] BATCH MODE: {len(tracks)} frames")
            batch_layers = []
            
            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                layer = self._render_single_frame(
                    frame_tracks, image_width, image_height, curve_type,
                    line_style, width_px, opacity, use_gpu, **kwargs
                )
                batch_layers.append(layer)
                print(f"[YS-LINE] Frame {i}: {len(frame_tracks)} tracks")
            
            # Stack into batch
            batch_result = np.stack(batch_layers, axis=0)
            print(f"[YS-LINE] Returning batch: {batch_result.shape}")
            # Don't use numpy_to_comfyui - already in BHWC format, just convert to tensor
            import torch
            return (torch.from_numpy(batch_result.astype(np.float32)),)
        
        # Single frame mode
        print(f"[YS-LINE] SINGLE MODE")
        layer = self._render_single_frame(
            tracks, image_width, image_height, curve_type,
            line_style, width_px, opacity, use_gpu, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, tracks, image_width, image_height, curve_type,
                            line_style, width_px, opacity, use_gpu, **kwargs):
        """Render single frame - extracted to avoid duplication"""

        # Convert tracks to numpy array
        if not isinstance(tracks, np.ndarray):
            tracks = np.array(tracks)

        if len(tracks) < 2:
            # Return empty layer
            layer = create_rgba_layer(image_height, image_width)
            return layer

        # Set random seed
        np.random.seed(kwargs.get('seed', 42))

        # Handle antialiasing
        aa_factor = self._get_aa_factor(kwargs.get('antialiasing', 'none'))
        if aa_factor > 1:
            layer = create_rgba_layer(image_height * aa_factor, image_width * aa_factor)
            tracks_scaled = tracks * aa_factor
        else:
            layer = create_rgba_layer(image_height, image_width)
            tracks_scaled = tracks

        # Initialize curve generator
        curve_gen = CurveGenerator(samples_per_curve=kwargs.get('samples_per_curve', 50))

        # Build graph connections
        edges = self._build_graph(tracks_scaled, curve_type, **kwargs)

        # Render each edge with specified curve type
        for i, (start_idx, end_idx) in enumerate(edges):
            p1 = tracks_scaled[start_idx]
            p2 = tracks_scaled[end_idx]

            # Generate curve points
            curve_points = self._generate_curve(curve_gen, p1, p2, curve_type, **kwargs)

            # Determine color
            color = self._get_edge_color(i, start_idx, end_idx, **kwargs)

            # Render with specified style
            self._render_line_styled(layer, curve_points, color,
                                    line_style, width_px * aa_factor,
                                    opacity, **kwargs)

        # Downscale if antialiasing was used
        if aa_factor > 1:
            layer = cv2.resize(layer, (image_width, image_height),
                             interpolation=cv2.INTER_AREA)

        return layer

    def _get_aa_factor(self, antialiasing: str) -> int:
        """Get antialiasing factor"""
        if antialiasing == '2x':
            return 2
        elif antialiasing == '4x':
            return 4
        return 1

    def _build_graph(self, tracks: np.ndarray, curve_type: str, **kwargs) -> List[Tuple[int, int]]:
        """Build graph connections with various strategies"""

        # For graph-based curve types, use their inherent structure
        if curve_type in ['delaunay', 'voronoi_edges', 'minimum_spanning']:
            if curve_type == 'delaunay':
                return GraphBuilder.build_delaunay_graph(tracks)
            elif curve_type == 'voronoi_edges':
                return GraphBuilder.build_voronoi_graph(tracks)
            elif curve_type == 'minimum_spanning':
                return GraphBuilder.build_mst_graph(tracks)

        # Otherwise use specified graph mode
        mode = kwargs.get('graph_mode', 'knn')

        if mode == 'knn':
            k = kwargs.get('k_neighbors', 3)
            return GraphBuilder.build_knn_graph(tracks, k)
        elif mode == 'radius':
            radius = kwargs.get('connection_radius', 100.0)
            return GraphBuilder.build_radius_graph(tracks, radius)
        elif mode == 'delaunay':
            return GraphBuilder.build_delaunay_graph(tracks)
        elif mode == 'mst':
            return GraphBuilder.build_mst_graph(tracks)
        elif mode == 'voronoi':
            return GraphBuilder.build_voronoi_graph(tracks)
        else:
            # Default to kNN
            return GraphBuilder.build_knn_graph(tracks, 3)

    def _generate_curve(self, curve_gen: CurveGenerator, p1: np.ndarray,
                       p2: np.ndarray, curve_type: str, **kwargs) -> np.ndarray:
        """Generate curve points using curve generator"""

        if curve_type == "straight":
            return curve_gen.generate_straight(p1, p2)

        elif curve_type == "quadratic_bezier":
            overshoot = kwargs.get('overshoot', 0.0)
            return curve_gen.generate_quadratic_bezier(p1, p2, overshoot)

        elif curve_type == "cubic_bezier":
            overshoot = kwargs.get('overshoot', 0.0)
            control_offset = kwargs.get('control_point_offset', 0.3)
            return curve_gen.generate_cubic_bezier(p1, p2, overshoot, control_offset)

        elif curve_type == "catmull_rom":
            tension = kwargs.get('curve_tension', 0.5)
            return curve_gen.generate_catmull_rom(p1, p2, tension)

        elif curve_type == "logarithmic_spiral":
            turns = kwargs.get('spiral_turns', 0.5)
            return curve_gen.generate_logarithmic_spiral(p1, p2, turns)

        elif curve_type == "elastic":
            stiffness = kwargs.get('elastic_stiffness', 0.5)
            return curve_gen.generate_elastic_curve(p1, p2, stiffness)

        elif curve_type == "fourier_series":
            return curve_gen.generate_fourier_series(p1, p2, n_harmonics=5)

        elif curve_type == "field_lines":
            field_strength = kwargs.get('field_strength', 1.0)
            return curve_gen.generate_field_lines(p1, p2, field_strength)

        elif curve_type == "gravitational":
            gravity = kwargs.get('gravity_strength', 0.1)
            return curve_gen.generate_gravitational_path(p1, p2, gravity)

        else:
            # Default to straight line
            return curve_gen.generate_straight(p1, p2)

    def _get_edge_color(self, edge_index: int, start_idx: int,
                       end_idx: int, **kwargs) -> np.ndarray:
        """Determine edge color"""

        # Parse fixed color from string
        color_str = kwargs.get('fixed_color', '1.0,1.0,1.0')
        try:
            color_parts = [float(x.strip()) for x in color_str.split(',')]
            if len(color_parts) >= 3:
                return np.array(color_parts[:3])
        except:
            pass

        # Default white
        return np.array([1.0, 1.0, 1.0])

    def _render_line_styled(self, layer: np.ndarray, points: np.ndarray,
                           color: np.ndarray, style: str, width: float,
                           opacity: float, **kwargs):
        """Render line with various styles"""

        if len(points) < 2:
            return

        if style == "solid":
            self._render_solid_line(layer, points, color, width, opacity)

        elif style == "dotted":
            spacing = kwargs.get('dot_spacing', 5.0)
            self._render_dotted_line(layer, points, color, width, opacity, spacing)

        elif style == "dashed":
            dash_length = kwargs.get('dash_length', 10.0)
            self._render_dashed_line(layer, points, color, width, opacity, dash_length)

        elif style == "dash_dot":
            dash_length = kwargs.get('dash_length', 10.0)
            self._render_dash_dot_line(layer, points, color, width, opacity, dash_length)

        elif style == "gradient_fade":
            start_color_str = kwargs.get('gradient_start_color', '1.0,1.0,1.0')
            end_color_str = kwargs.get('gradient_end_color', '0.0,0.5,1.0')

            try:
                start_color = np.array([float(x.strip()) for x in start_color_str.split(',')])[:3]
                end_color = np.array([float(x.strip()) for x in end_color_str.split(',')])[:3]
            except:
                start_color = color
                end_color = color

            self._render_gradient_line(layer, points, start_color, end_color, width, opacity)

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
            n_particles = max(10, int(len(points) / 3))
            particle_indices = np.random.choice(len(points), min(n_particles, len(points)), replace=False)

            for idx in particle_indices:
                pt = points[idx]
                size = np.random.uniform(width * 0.5, width * 2)
                self._draw_soft_circle(layer, pt[0], pt[1], size, color, opacity * 0.7)

        elif style == "wave":
            # Sinusoidal modulation
            amplitude = kwargs.get('wave_amplitude', 5.0)
            frequency = kwargs.get('wave_frequency', 0.1)

            # Add wave perpendicular to path
            wave_points = []
            for i, pt in enumerate(points):
                if i > 0:
                    tangent = points[i] - points[i-1]
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 0:
                        normal = np.array([-tangent[1], tangent[0]]) / tangent_norm
                        offset = amplitude * np.sin(i * frequency)
                        wave_pt = pt + normal * offset
                    else:
                        wave_pt = pt
                else:
                    wave_pt = pt

                wave_points.append(wave_pt)

            self._render_solid_line(layer, np.array(wave_points), color, width, opacity)

        else:
            # Default to solid
            self._render_solid_line(layer, points, color, width, opacity)

    def _render_solid_line(self, layer: np.ndarray, points: np.ndarray,
                          color: np.ndarray, width: float, opacity: float):
        """Render solid line with anti-aliasing"""

        # Convert to integer coordinates
        points_int = points.astype(np.int32)

        # Clip to layer bounds
        h, w = layer.shape[:2]

        # Draw using OpenCV with anti-aliasing
        for i in range(len(points_int) - 1):
            pt1 = tuple(points_int[i])
            pt2 = tuple(points_int[i+1])

            # Create RGBA color
            rgba = np.concatenate([color, [opacity]])

            # Draw line segment
            cv2.line(layer, pt1, pt2, tuple(rgba), int(np.ceil(width)), cv2.LINE_AA)

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
                rgba = np.concatenate([color, [opacity]])
                cv2.circle(layer, tuple(pt.astype(int)), int(width), tuple(rgba), -1, cv2.LINE_AA)

    def _render_dashed_line(self, layer: np.ndarray, points: np.ndarray,
                           color: np.ndarray, width: float, opacity: float,
                           dash_length: float):
        """Render dashed line"""

        # Calculate total line length
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]

        # Draw dashes
        current_dist = 0
        draw_dash = True

        while current_dist < total_length:
            if draw_dash:
                # Draw dash
                start_dist = current_dist
                end_dist = min(current_dist + dash_length, total_length)

                # Find start point
                start_idx = np.searchsorted(cumulative, start_dist)
                if start_idx > 0 and start_idx < len(points):
                    t_start = (start_dist - cumulative[start_idx-1]) / (distances[start_idx-1] + 1e-6)
                    pt_start = points[start_idx-1] * (1-t_start) + points[start_idx] * t_start
                else:
                    pt_start = points[0]

                # Find end point
                end_idx = np.searchsorted(cumulative, end_dist)
                if end_idx > 0 and end_idx < len(points):
                    t_end = (end_dist - cumulative[end_idx-1]) / (distances[end_idx-1] + 1e-6)
                    pt_end = points[end_idx-1] * (1-t_end) + points[end_idx] * t_end
                else:
                    pt_end = points[-1]

                # Draw dash segment
                rgba = np.concatenate([color, [opacity]])
                cv2.line(layer, tuple(pt_start.astype(int)), tuple(pt_end.astype(int)),
                        tuple(rgba), int(np.ceil(width)), cv2.LINE_AA)

            current_dist += dash_length
            draw_dash = not draw_dash

    def _render_dash_dot_line(self, layer: np.ndarray, points: np.ndarray,
                             color: np.ndarray, width: float, opacity: float,
                             dash_length: float):
        """Render dash-dot line"""
        # Simplified: render as dashed for now
        self._render_dashed_line(layer, points, color, width, opacity, dash_length)

    def _render_gradient_line(self, layer: np.ndarray, points: np.ndarray,
                             start_color: np.ndarray, end_color: np.ndarray,
                             width: float, opacity: float):
        """Render line with color gradient"""

        for i in range(len(points) - 1):
            # Interpolate color
            t = i / (len(points) - 1)
            color = start_color * (1-t) + end_color * t

            # Draw segment
            rgba = np.concatenate([color, [opacity]])
            cv2.line(layer, tuple(points[i].astype(int)),
                    tuple(points[i+1].astype(int)),
                    tuple(rgba), int(np.ceil(width)), cv2.LINE_AA)

    def _draw_soft_circle(self, layer: np.ndarray, x: float, y: float,
                         radius: float, color: np.ndarray, opacity: float):
        """Draw soft circle (for particle effect)"""

        rgba = np.concatenate([color, [opacity]])
        cv2.circle(layer, (int(x), int(y)), int(radius), tuple(rgba), -1, cv2.LINE_AA)


# ComfyUI node registration information
NODE_CLASS_MAPPINGS = {
    "YS_LineLinkRenderer": AdvancedLineLinkRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_LineLinkRenderer": "Line Link Renderer (Advanced) 🌀"
}