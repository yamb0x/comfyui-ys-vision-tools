"""
Graph Line Renderer Node for YS-vision-tools
Specialized node for network/graph-based line connections

🎨 COLOR PICKER SUPPORT:
- Visual color picker UI (click the color swatch in ComfyUI)
- Supports HEX colors: "#ffffff", "#ff0000", "#00ff00"
- Supports named colors: "red", "orange", "cyan", "white"
- Backward compatible with legacy float lists: [1.0, 0.5, 0.0]
- Separate alpha slider for transparency control
"""

import numpy as np
from typing import Dict, Any

from .line_link_renderer import AdvancedLineLinkRendererNode


class GraphLineRendererNode:
    """Render graph-based network lines with preset patterns"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),

                "network_type": ([
                    "neural_web",       # Delaunay triangulation for neural network look
                    "organic_mesh",     # Voronoi edges for organic patterns
                    "minimal_tree",     # Minimum spanning tree for minimal connections
                    "proximity_net",    # Radius-based connections
                ], {
                    "default": "neural_web",
                    "tooltip": "Network connection pattern"
                }),

                "line_style": ([
                    "solid",
                    "dotted",
                    "dashed",
                    "gradient",
                    "wave",
                ], {
                    "default": "solid"
                }),

                "line_thickness": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color": ("COLOR", {
                    "default": "#ffffff",
                    "tooltip": "Click the color swatch to open the visual color picker"
                }),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Transparency level (0=invisible, 1=opaque)"
                }),
                "connection_distance": ("FLOAT", {
                    "default": 150.0,
                    "min": 50.0,
                    "max": 500.0,
                    "step": 10.0,
                    "tooltip": "Maximum connection distance (for proximity_net)"
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering/Simplified"

    def __init__(self):
        self.advanced_node = AdvancedLineLinkRendererNode()

    def execute(self, tracks, image_width, image_height, network_type, line_style,
                line_thickness, opacity, color, **kwargs):
        """Render graph-based network lines
        
        Args:
            color: COLOR input (hex/named color)
            **kwargs: Optional parameters including 'alpha'
        """

        # Map line_style names
        style_map = {
            "solid": "solid",
            "dotted": "dotted",
            "dashed": "dashed",
            "gradient": "gradient_fade",
            "wave": "wave"
        }

        # Map network_type to curve_type and graph_mode
        network_configs = {
            "neural_web": {
                "curve_type": "straight",
                "graph_mode": "delaunay",
            },
            "organic_mesh": {
                "curve_type": "straight",
                "graph_mode": "voronoi",
            },
            "minimal_tree": {
                "curve_type": "straight",
                "graph_mode": "mst",
            },
            "proximity_net": {
                "curve_type": "straight",
                "graph_mode": "radius",
                "connection_radius": kwargs.get('connection_distance', 150.0),
            },
        }

        config = network_configs.get(network_type, network_configs["neural_web"])

        # Extract graph-specific parameters (not curve_type)
        graph_params = {k: v for k, v in config.items() if k != "curve_type"}

        # Build kwargs for advanced node
        advanced_kwargs = {
            "antialiasing": "2x",
            "samples_per_curve": 30,
            "color": color,  # Pass COLOR input to advanced node
            **graph_params,
            **kwargs
        }

        # Call advanced node
        return self.advanced_node.execute(
            tracks=tracks,
            image_width=image_width,
            image_height=image_height,
            curve_type=config["curve_type"],
            line_style=style_map[line_style],
            width_px=line_thickness,
            opacity=opacity,
            use_gpu=kwargs.get('use_gpu', True),
            **advanced_kwargs
        )


NODE_CLASS_MAPPINGS = {
    "YS_GraphLineRenderer": GraphLineRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_GraphLineRenderer": "Graph Line Renderer 🕸️"
}