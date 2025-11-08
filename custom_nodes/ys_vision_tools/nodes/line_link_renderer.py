"""
Advanced Line Link Renderer Node for YS-vision-tools
Renders sophisticated curves with 15+ mathematical types and 10+ animated styles

🎨 COLOR PICKER SUPPORT:
- Visual color picker UI for gradient colors (gradient_fade style)
- Supports HEX colors: "#ffffff", "#ff0000", "#00ff00"
- Supports named colors: "red", "orange", "cyan", "white"
- Backward compatible with legacy float lists: [1.0, 0.5, 0.0]
- Separate gradient_alpha slider for transparency control
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Dict, Any

from ..utils import (
    get_gpu_accelerator,
    is_gpu_available,
    CurveGenerator,
    GraphBuilder,
    GPUCurveBatchGenerator,
    CUPY_AVAILABLE,
    create_rgba_layer,
    numpy_to_comfyui,
    normalize_color_to_rgba01
)

# GPU imports
try:
    import cupy as cp
    # CUPY_AVAILABLE already imported from utils
except ImportError:
    cp = None

# GPU graph builder
try:
    from ..utils.gpu_graph import GPUGraphBuilder, GPU_GRAPH_AVAILABLE
except ImportError:
    GPU_GRAPH_AVAILABLE = False
    GPUGraphBuilder = None


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
                    "catmull_rom",
                    "logarithmic_spiral",
                    "elastic",
                    "fourier_series",
                    "field_lines",
                    "gravitational",
                    "delaunay",
                    "voronoi_edges",
                    "minimum_spanning",
                ], {
                    "tooltip": "Curve type:\n"
                               "• straight - Direct line (no params)\n"
                               "• quadratic_bezier - Uses: overshoot\n"
                               "• cubic_bezier - Uses: overshoot, control_point_offset\n"
                               "• catmull_rom - Uses: curve_tension\n"
                               "• logarithmic_spiral - Uses: spiral_turns\n"
                               "• elastic - Uses: elastic_stiffness\n"
                               "• fourier_series - No params (5 harmonics)\n"
                               "• field_lines - Uses: field_strength\n"
                               "• gravitational - Uses: gravity_strength\n"
                               "• delaunay/voronoi_edges/minimum_spanning - Graph-based (overrides graph_mode)"
                }),

                "line_style": ([
                    "solid",
                    "dotted",
                    "dashed",
                    "dash_dot",
                    "gradient_fade",
                    "pulsing",
                    "electric",
                    "particle_trail",
                    "wave",
                ], {
                    "tooltip": "Line style:\n"
                               "• solid - No params\n"
                               "• dotted - Uses: dot_spacing\n"
                               "• dashed - Uses: dash_length\n"
                               "• dash_dot - Uses: dash_length\n"
                               "• gradient_fade - Uses: gradient_start_color, gradient_end_color\n"
                               "• pulsing - Uses: pulse_frequency, time\n"
                               "• electric - No params (randomized)\n"
                               "• particle_trail - No params (randomized)\n"
                               "• wave - Uses: wave_amplitude, wave_frequency"
                }),

                "width_px": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 20.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color": ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Line color (click the color swatch to open the visual color picker)"
                }),
                "use_gpu": ("BOOLEAN", {"default": True, "tooltip": "Enable GPU acceleration for graph building and curve generation"}),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Transparency level (0=invisible, 1=opaque)"
                }),
                # Graph construction
                "graph_mode": (["knn", "radius", "delaunay", "mst", "voronoi"], {
                    "tooltip": "Graph connection strategy. Ignored when curve_type is delaunay/voronoi_edges/minimum_spanning"
                }),
                "k_neighbors": ("INT", {
                    "default": 3, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Number of nearest neighbors. Only used with graph_mode=knn"
                }),
                "connection_radius": ("FLOAT", {
                    "default": 100.0, "min": 10.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Connection radius in pixels. Only used with graph_mode=radius"
                }),
                
                # GPU Graph Optimization
                "delta_y_max": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2000.0,
                    "step": 10.0,
                    "tooltip": "GPU-only: Max vertical distance for connections (0=disabled). Requires use_gpu=True and FAISS-GPU"
                }),
                "degree_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "GPU-only: Max connections per point (0=disabled). Requires use_gpu=True and FAISS-GPU"
                }),
                "hysteresis_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.1,
                    "tooltip": "GPU-only: Smooth edge changes between frames (0=disabled, 0.5=recommended). Requires use_gpu=True and FAISS-GPU"
                }),

                # Curve parameters
                "curve_tension": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Only used by: catmull_rom curve"
                }),
                "overshoot": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used by: quadratic_bezier, cubic_bezier curves"
                }),
                "control_point_offset": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used by: cubic_bezier curve"
                }),
                "spiral_turns": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Only used by: logarithmic_spiral curve"
                }),

                # Style parameters
                "dot_spacing": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Only used by: dotted line style"
                }),
                "dash_length": ("FLOAT", {
                    "default": 10.0, "min": 2.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Only used by: dashed, dash_dot line styles"
                }),
                "gradient_start_color": ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Gradient start color (only for gradient_fade style)"
                }),
                "gradient_end_color": ("COLOR", {
                    "default": "#000000",
                    "tooltip": "Gradient end color (only for gradient_fade style)"
                }),
                "gradient_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Transparency for gradient (only for gradient_fade style)"
                }),
                "pulse_frequency": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Only used by: pulsing line style"
                }),
                "wave_amplitude": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Only used by: wave line style"
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Only used by: wave line style"
                }),

                # Physics parameters
                "gravity_strength": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only used by: gravitational curve"
                }),
                "elastic_stiffness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Only used by: elastic curve"
                }),
                "field_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Only used by: field_lines curve"
                }),

                # Animation
                "time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Animation time parameter. Used by: pulsing line style"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffff, "step": 1,
                    "tooltip": "Random seed for: electric, particle_trail line styles"
                }),

                # Performance
                "samples_per_curve": ("INT", {
                    "default": 50, "min": 10, "max": 200, "step": 10,
                    "tooltip": "Number of points per curve (higher = smoother but slower)"
                }),
                "antialiasing": (["none", "2x", "4x"], {
                    "tooltip": "Supersampling antialiasing (2x/4x = better quality but slower)"
                }),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering"

    def __init__(self):
        self.gpu = get_gpu_accelerator()
        np.random.seed(42)
        
        # Initialize GPU graph builder
        self.gpu_graph_builder = None
        if GPU_GRAPH_AVAILABLE:
            try:
                self.gpu_graph_builder = GPUGraphBuilder()
                print("[YS-LINE] GPU graph builder initialized (FAISS-GPU available)")
            except Exception as e:
                print(f"[YS-LINE] GPU graph builder init failed: {e}")
                self.gpu_graph_builder = None
        else:
            print("[YS-LINE] GPU graph builder not available, using CPU fallback")
        
        # Initialize GPU curve batch generator
        self.gpu_curve_gen = None
        if CUPY_AVAILABLE:
            try:
                # Will be created with proper samples_per_curve in _render_single_frame
                print("[YS-LINE] GPU curve generation available (CuPy detected)")
            except Exception as e:
                print(f"[YS-LINE] GPU curve init check failed: {e}")
        else:
            print("[YS-LINE] GPU curve generation not available, using CPU fallback")

    def execute(self, tracks, image_width, image_height, curve_type,
                line_style, width_px, opacity, color, use_gpu, **kwargs):
        """
        Render advanced lines with experimental curves.

        Args:
            color: COLOR input (hex string like "#ffffff" or named color like "red")
                   Automatically parsed by normalize_color_to_rgba01()
            **kwargs: Optional parameters including 'alpha' for transparency
        """

        # DEBUG
        print(f"\n[YS-LINE] Executing LineLinkRenderer")
        print(f"[YS-LINE] color input: {color} (type: {type(color)})")
        print(f"[YS-LINE] tracks type: {type(tracks)}")
        
        # Check if batch mode (list of track arrays)
        if isinstance(tracks, list):
            print(f"[YS-LINE] BATCH MODE: {len(tracks)} frames")
            batch_layers = []
            
            for i, frame_tracks in enumerate(tracks):
                # Process single frame
                layer = self._render_single_frame(
                    frame_tracks, image_width, image_height, curve_type,
                    line_style, width_px, opacity, color, use_gpu, **kwargs
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
            line_style, width_px, opacity, color, use_gpu, **kwargs
        )
        return (numpy_to_comfyui(layer),)

    def _render_single_frame(self, tracks, image_width, image_height, curve_type,
                            line_style, width_px, opacity, color, use_gpu, **kwargs):
        """Render single frame - extracted to avoid duplication

        Args:
            color: COLOR input (hex/named color) passed from execute()
            **kwargs: Optional parameters including 'alpha' override
        """

        # Parse color using centralized utility
        # Alpha from optional parameter, defaults to opacity if not provided
        alpha = kwargs.get('alpha', opacity)
        rgba = normalize_color_to_rgba01(color, alpha)
        color_rgb = np.array(rgba[:3])  # Extract RGB as numpy array for rendering functions

        print(f"[YS-LINE] Parsed color: {color} -> RGBA: {rgba}")

        # Add parsed color to kwargs for downstream methods
        kwargs['color_rgb'] = color_rgb

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

        # Build graph connections
        edges = self._build_graph(tracks_scaled, curve_type, use_gpu, **kwargs)
        
        if len(edges) == 0:
            return layer

        # Try GPU batch curve generation first
        samples_per_curve = kwargs.get('samples_per_curve', 50)
        gpu_curves_generated = False
        
        # Check if curve type supports GPU batch generation
        gpu_supported_curves = [
            'straight', 'quadratic_bezier', 'cubic_bezier', 'catmull_rom',
            'logarithmic_spiral', 'elastic', 'gravitational'
        ]
        
        if use_gpu and CUPY_AVAILABLE and curve_type in gpu_supported_curves:
            try:
                # Generate ALL curves on GPU in one batch
                all_curves = self._generate_curves_batch_gpu(
                    tracks_scaled, edges, curve_type, **kwargs
                )
                gpu_curves_generated = True
                
                # Render each curve (rendering still on CPU for now)
                for i, curve_points in enumerate(all_curves):
                    start_idx, end_idx = edges[i]
                    color = self._get_edge_color(i, start_idx, end_idx, **kwargs)
                    self._render_line_styled(layer, curve_points, color,
                                           line_style, width_px * aa_factor,
                                           opacity, **kwargs)
                
            except Exception as e:
                error_msg = str(e)
                if "cl.exe" in error_msg or "nvcc" in error_msg:
                    print(f"[YS-LINE] GPU compilation unavailable (missing MSVC compiler), using CPU fallback")
                else:
                    print(f"[YS-LINE] GPU batch curve generation failed: {e}")
                print(f"[YS-LINE] Falling back to CPU curve generation")
                gpu_curves_generated = False
        
        # CPU fallback: loop-based generation
        if not gpu_curves_generated:
            curve_gen = CurveGenerator(samples_per_curve=samples_per_curve)
            
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

    def _build_graph(self, tracks: np.ndarray, curve_type: str, use_gpu: bool = True, **kwargs) -> List[Tuple[int, int]]:
        """Build graph connections with various strategies (GPU-accelerated when available)"""

        # For graph-based curve types, use their inherent structure
        if curve_type in ['delaunay', 'voronoi_edges', 'minimum_spanning']:
            # Inform user if graph_mode was specified but will be ignored
            if 'graph_mode' in kwargs:
                graph_names = {
                    'delaunay': 'Delaunay triangulation',
                    'voronoi_edges': 'Voronoi diagram',
                    'minimum_spanning': 'Minimum Spanning Tree'
                }
                print(f"[YS-LINE] Note: curve_type='{curve_type}' uses {graph_names.get(curve_type, 'built-in')} graph (graph_mode parameter not used)")
            
            if curve_type == 'delaunay':
                return GraphBuilder.build_delaunay_graph(tracks)
            elif curve_type == 'voronoi_edges':
                return GraphBuilder.build_voronoi_graph(tracks)
            elif curve_type == 'minimum_spanning':
                return GraphBuilder.build_mst_graph(tracks)

        # Otherwise use specified graph mode
        mode = kwargs.get('graph_mode', 'knn')

        # GPU-accelerated kNN path
        if mode == 'knn':
            k = kwargs.get('k_neighbors', 3)
            delta_y_max = kwargs.get('delta_y_max', 0.0)
            degree_cap = kwargs.get('degree_cap', 0)
            hysteresis_alpha = kwargs.get('hysteresis_alpha', 0.0)
            
            # Warn if GPU-specific params are used but GPU unavailable
            gpu_params_used = delta_y_max > 0 or degree_cap > 0 or hysteresis_alpha > 0
            if gpu_params_used and (not use_gpu or self.gpu_graph_builder is None):
                ignored_params = []
                if delta_y_max > 0:
                    ignored_params.append(f"delta_y_max={delta_y_max}")
                if degree_cap > 0:
                    ignored_params.append(f"degree_cap={degree_cap}")
                if hysteresis_alpha > 0:
                    ignored_params.append(f"hysteresis_alpha={hysteresis_alpha}")
                
                reason = "GPU disabled" if not use_gpu else "FAISS-GPU unavailable"
                print(f"[YS-LINE] Warning: GPU-only params ignored ({reason}): {', '.join(ignored_params)}")
            
            # GPU path: FAISS-GPU kNN
            if use_gpu and self.gpu_graph_builder is not None:
                try:
                    start_time = time.perf_counter()
                    
                    edges = self.gpu_graph_builder.build_knn_graph_gpu(
                        tracks,
                        k=k,
                        delta_y_max=delta_y_max if delta_y_max > 0 else None,
                        degree_cap=degree_cap if degree_cap > 0 else None,
                        hysteresis_alpha=hysteresis_alpha
                    )
                    
                    gpu_time = (time.perf_counter() - start_time) * 1000
                    print(f"[YS-LINE] GPU graph built: {len(tracks)} points → {len(edges)} edges in {gpu_time:.2f}ms")
                    
                    return edges
                    
                except Exception as e:
                    print(f"[YS-LINE] GPU graph building failed: {e}, falling back to CPU")
            
            # CPU fallback
            start_time = time.perf_counter()
            edges = GraphBuilder.build_knn_graph(tracks, k)
            cpu_time = (time.perf_counter() - start_time) * 1000
            print(f"[YS-LINE] CPU graph built: {len(tracks)} points → {len(edges)} edges in {cpu_time:.2f}ms")
            
            # Warn if k_neighbors used with non-knn mode was expected but we're in fallback
            return edges
            
        elif mode == 'radius':
            radius = kwargs.get('connection_radius', 100.0)
            # Warn if k_neighbors specified but not used
            if 'k_neighbors' in kwargs and kwargs['k_neighbors'] != 3:
                print(f"[YS-LINE] Warning: k_neighbors ignored - graph_mode='radius' uses connection_radius instead")
            return GraphBuilder.build_radius_graph(tracks, radius)
        elif mode == 'delaunay':
            return GraphBuilder.build_delaunay_graph(tracks)
        elif mode == 'mst':
            return GraphBuilder.build_mst_graph(tracks)
        elif mode == 'voronoi':
            return GraphBuilder.build_voronoi_graph(tracks)
        else:
            # Default to kNN
            k = kwargs.get('k_neighbors', 3)
            return GraphBuilder.build_knn_graph(tracks, k)

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

    def _generate_curves_batch_gpu(self, tracks: np.ndarray, edges: List[Tuple[int, int]],
                                   curve_type: str, **kwargs) -> List[np.ndarray]:
        """
        Generate ALL curves in batch on GPU (massive speedup!)

        Args:
            tracks: All track points (N_points, 2)
            edges: List of (start_idx, end_idx) tuples
            curve_type: Type of curve to generate
            **kwargs: Curve-specific parameters (including samples_per_curve)

        Returns:
            List of curve point arrays (each is (samples, 2))
        """
        if not CUPY_AVAILABLE or cp is None:
            raise RuntimeError("CuPy not available for GPU curve generation")

        # Extract samples_per_curve from kwargs
        samples_per_curve = kwargs.get('samples_per_curve', 100)

        start_time = time.perf_counter()
        n_edges = len(edges)
        
        # Extract start and end points for all edges
        edges_array = np.array(edges)
        p1_batch = tracks[edges_array[:, 0]]  # (N_edges, 2)
        p2_batch = tracks[edges_array[:, 1]]  # (N_edges, 2)
        
        # Transfer to GPU
        p1_gpu = cp.asarray(p1_batch, dtype=cp.float32)
        p2_gpu = cp.asarray(p2_batch, dtype=cp.float32)
        
        # Create GPU batch generator
        gpu_gen = GPUCurveBatchGenerator(samples_per_curve=samples_per_curve)
        
        # Generate curves based on type
        if curve_type == "straight":
            curves_gpu = gpu_gen.generate_straight_batch(p1_gpu, p2_gpu)
        
        elif curve_type == "quadratic_bezier":
            overshoot = kwargs.get('overshoot', 0.0)
            curves_gpu = gpu_gen.generate_quadratic_bezier_batch(p1_gpu, p2_gpu, overshoot)
        
        elif curve_type == "cubic_bezier":
            overshoot = kwargs.get('overshoot', 0.0)
            control_offset = kwargs.get('control_point_offset', 0.3)
            curves_gpu = gpu_gen.generate_cubic_bezier_batch(p1_gpu, p2_gpu, overshoot, control_offset)
        
        elif curve_type == "catmull_rom":
            tension = kwargs.get('curve_tension', 0.5)
            curves_gpu = gpu_gen.generate_catmull_rom_batch(p1_gpu, p2_gpu, tension)
        
        elif curve_type == "logarithmic_spiral":
            turns = kwargs.get('spiral_turns', 0.5)
            curves_gpu = gpu_gen.generate_logarithmic_spiral_batch(p1_gpu, p2_gpu, turns)
        
        elif curve_type == "elastic":
            stiffness = kwargs.get('elastic_stiffness', 0.5)
            curves_gpu = gpu_gen.generate_elastic_batch(p1_gpu, p2_gpu, stiffness)
        
        elif curve_type == "gravitational":
            gravity = kwargs.get('gravity_strength', 0.1)
            curves_gpu = gpu_gen.generate_gravitational_batch(p1_gpu, p2_gpu, gravity)
        
        else:
            # Fallback to straight
            curves_gpu = gpu_gen.generate_straight_batch(p1_gpu, p2_gpu)
        
        # Transfer back to CPU
        curves_cpu = cp.asnumpy(curves_gpu)  # (N_edges, samples, 2)
        
        gpu_time = (time.perf_counter() - start_time) * 1000
        print(f"[YS-LINE] GPU generated {n_edges} {curve_type} curves in {gpu_time:.2f}ms ({gpu_time/n_edges:.3f}ms per curve)")
        
        # Convert to list of individual curve arrays
        curves_list = [curves_cpu[i] for i in range(n_edges)]
        return curves_list

    def _get_edge_color(self, edge_index: int, start_idx: int,
                       end_idx: int, **kwargs) -> np.ndarray:
        """Determine edge color"""

        # Use parsed color_rgb from kwargs (set in _render_single_frame)
        color_rgb = kwargs.get('color_rgb', np.array([1.0, 1.0, 1.0]))
        return color_rgb

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
            # Parse gradient colors using centralized utility
            gradient_alpha = kwargs.get('gradient_alpha', opacity)
            start_rgba = normalize_color_to_rgba01(
                kwargs.get('gradient_start_color', '#ffffff'),
                gradient_alpha
            )
            end_rgba = normalize_color_to_rgba01(
                kwargs.get('gradient_end_color', '#000000'),
                gradient_alpha
            )
            start_color = start_rgba[:3]
            end_color = end_rgba[:3]

            print(f"[YS-LINE] Parsed gradient: {kwargs.get('gradient_start_color')} -> {start_rgba}, {kwargs.get('gradient_end_color')} -> {end_rgba}")

            self._render_gradient_line(layer, points, start_color, end_color, width, gradient_alpha)

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