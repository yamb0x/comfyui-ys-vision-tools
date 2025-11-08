"""
Physics Line Renderer Node for YS-vision-tools
Specialized node for physics-simulated line connections

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


class PhysicsLineRendererNode:
    """Render physics-simulated lines with force fields and energy"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "tracks": ("TRACKS",),
                "image_width": ("INT", {"default": 1920, "min": 64, "max": 7680, "step": 1}),
                "image_height": ("INT", {"default": 1080, "min": 64, "max": 4320, "step": 1}),

                "physics_type": ([
                    "electric_field",   # Field lines like electromagnetic fields
                    "gravity_pull",     # Gravitational attraction curves
                    "particle_swarm",   # Particle trail effects
                    "energy_waves",     # Wave propagation effects
                ], {
                    "default": "electric_field",
                    "tooltip": "Physics simulation type"
                }),

                "effect_style": ([
                    "electric",         # Lightning-like electric effect
                    "pulsing",         # Animated pulsing
                    "particle_trail",  # Particle system trail
                    "gradient",        # Smooth gradient fade
                ], {
                    "default": "electric"
                }),

                "line_thickness": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "connections": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of connections per point"
                }),
                "color": ("COLOR", {
                    "default": "#4db3ff",
                    "tooltip": "Click the color swatch to open the visual color picker (default: electric blue)"
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
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Physics effect intensity"
                }),
                "animation_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Animation time for pulsing/wave effects"
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LAYER",)
    FUNCTION = "execute"
    CATEGORY = "YS-vision-tools/Rendering/Simplified"

    def __init__(self):
        self.advanced_node = AdvancedLineLinkRendererNode()

    def execute(self, tracks, image_width, image_height, physics_type, effect_style,
                line_thickness, opacity, connections, color, **kwargs):
        """Render physics-simulated lines
        
        Args:
            color: COLOR input (hex/named color)
            **kwargs: Optional parameters including 'alpha'
        """

        # Map effect_style names
        style_map = {
            "electric": "electric",
            "pulsing": "pulsing",
            "particle_trail": "particle_trail",
            "gradient": "gradient_fade"
        }

        # Map physics_type to curve_type and parameters
        physics_configs = {
            "electric_field": {
                "curve_type": "field_lines",
                "field_strength": kwargs.get('intensity', 1.0),
            },
            "gravity_pull": {
                "curve_type": "gravitational",
                "gravity_strength": kwargs.get('intensity', 1.0) * 0.3,
            },
            "particle_swarm": {
                "curve_type": "elastic",
                "elastic_stiffness": kwargs.get('intensity', 1.0) * 0.5,
            },
            "energy_waves": {
                "curve_type": "fourier_series",
                "wave_amplitude": kwargs.get('intensity', 1.0) * 10.0,
                "wave_frequency": 0.15,
            },
        }

        config = physics_configs.get(physics_type, physics_configs["electric_field"])

        # Extract physics-specific parameters (not curve_type)
        physics_params = {k: v for k, v in config.items() if k != "curve_type"}

        # Build kwargs for advanced node
        advanced_kwargs = {
            "graph_mode": "knn",
            "k_neighbors": connections,
            "antialiasing": "2x",
            "samples_per_curve": 60,
            "color": color,  # Pass COLOR input to advanced node
            "time": kwargs.get('animation_time', 0.0),
            **physics_params,
            **kwargs
        }

        # Add pulse frequency for pulsing style
        if effect_style == "pulsing":
            advanced_kwargs["pulse_frequency"] = 2.0

        # Call advanced node
        return self.advanced_node.execute(
            tracks=tracks,
            image_width=image_width,
            image_height=image_height,
            curve_type=config["curve_type"],
            line_style=style_map[effect_style],
            width_px=line_thickness,
            opacity=opacity,
            use_gpu=kwargs.get('use_gpu', True),
            **advanced_kwargs
        )


NODE_CLASS_MAPPINGS = {
    "YS_PhysicsLineRenderer": PhysicsLineRendererNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YS_PhysicsLineRenderer": "Physics Line Renderer ⚡"
}