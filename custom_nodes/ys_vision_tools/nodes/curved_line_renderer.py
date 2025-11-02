"""
Curved Line Renderer Node for YS-vision-tools
Specialized node for smooth, curved line connections with preset styles
"""

import numpy as np
from typing import Dict, Any

from .line_link_renderer import AdvancedLineLinkRendererNode


class CurvedLineRendererNode:
    """Render smooth curved lines with preset styles - simplified interface"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),

                "style": ([
                    "smooth_flow",      # Catmull-Rom curves, gradient fade
                    "elastic_bounce",   # Elastic physics curves
                    "spiral_energy",    # Logarithmic spirals
                    "bezier_smooth",    # Cubic Bezier curves
                ], {
                    "default": "smooth_flow",
                    "tooltip": "Preset curve styles with different mathematical curves"
                }),

                "line_thickness": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "connections": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of nearest neighbors to connect"
                }),
            },
            "optional": {
                "color": ("STRING", {"default": "1.0,1.0,1.0"}),
                "smoothness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Curve tension/smoothness"
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering/Simplified"

    def __init__(self):
        self.advanced_node = AdvancedLineLinkRendererNode()

    def execute(self, tracks, image_width, image_height, style, line_thickness,
                opacity, connections, **kwargs):
        """Render curved lines with preset configurations"""

        # Map style to advanced node parameters
        style_configs = {
            "smooth_flow": {
                "curve_type": "catmull_rom",
                "line_style": "gradient_fade",
                "curve_tension": kwargs.get('smoothness', 0.5),
            },
            "elastic_bounce": {
                "curve_type": "elastic",
                "line_style": "solid",
                "elastic_stiffness": kwargs.get('smoothness', 0.5),
            },
            "spiral_energy": {
                "curve_type": "logarithmic_spiral",
                "line_style": "gradient_fade",
                "spiral_turns": kwargs.get('smoothness', 0.5) * 2.0,
            },
            "bezier_smooth": {
                "curve_type": "cubic_bezier",
                "line_style": "solid",
                "overshoot": kwargs.get('smoothness', 0.5) * 0.6 - 0.3,
            },
        }

        config = style_configs.get(style, style_configs["smooth_flow"])

        # Extract curve-specific parameters (not curve_type/line_style)
        curve_params = {k: v for k, v in config.items() if k not in ["curve_type", "line_style"]}

        # Build kwargs for advanced node
        advanced_kwargs = {
            "graph_mode": "knn",
            "k_neighbors": connections,
            "antialiasing": "2x",
            "samples_per_curve": 50,
            "fixed_color": kwargs.get('color', '1.0,1.0,1.0'),
            **curve_params
        }

        # Call advanced node
        return self.advanced_node.execute(
            tracks=tracks,
            image_width=image_width,
            image_height=image_height,
            curve_type=config["curve_type"],
            line_style=config["line_style"],
            width_px=line_thickness,
            opacity=opacity,
            use_gpu=kwargs.get('use_gpu', True),
            **advanced_kwargs
        )


NODE_CLASS_MAPPINGS = {
    "YS_CurvedLineRenderer": CurvedLineRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_CurvedLineRenderer": "Curved Line Renderer 〰️"
}
