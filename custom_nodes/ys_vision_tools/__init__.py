"""
YS-vision-tools: Advanced ComfyUI custom nodes for GPU-accelerated vision overlays

A sophisticated visual effects system featuring:
- 7+ smart detection methods (gradient, phase, optical flow, YOLO, etc.)
- 15+ mathematical curve types (spirals, field lines, Fourier, etc.)
- 10+ animated line styles (electric, pulsing, wave, particle trails)
- GPU-first architecture optimized for RTX 5090 at 4K@60fps

Author: Yambo Studio
Target Platform: NVIDIA RTX 5090 (24GB VRAM)
Performance: 4K @ 60fps, 8K @ 30fps
"""

from .utils import (
    GPUAccelerator,
    get_gpu_accelerator,
    is_gpu_available,
    GPU_AVAILABLE
)

__version__ = "0.1.0"
__author__ = "Yambo Studio"

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import and register all nodes
# Phase 1 Nodes
from .nodes.track_detect import EnhancedTrackDetectNode
from .nodes.line_link_renderer import AdvancedLineLinkRendererNode
from .nodes.dot_renderer import DotRendererNode
from .nodes.palette_map import PaletteMapNode
from .nodes.layer_merge import LayerMergeNode
from .nodes.composite_over import CompositeOverNode

# Phase 1.5 Nodes (UX & Video Support)
from .nodes.image_size_detector import ImageSizeDetectorNode
from .nodes.video_frame_offset import VideoFrameOffsetNode

# Phase 2 Nodes (Extended Renderers)
from .nodes.bbox_renderer import BoundingBoxRendererNode
from .nodes.blur_region_renderer import BlurRegionRendererNode
from .nodes.mv_look_renderer import MVLookRendererNode

# Simplified Line Renderers (Preset-based)
from .nodes.curved_line_renderer import CurvedLineRendererNode
from .nodes.graph_line_renderer import GraphLineRendererNode
from .nodes.physics_line_renderer import PhysicsLineRendererNode

# Phase 1: Register Track Detect
NODE_CLASS_MAPPINGS["YS_TrackDetect"] = EnhancedTrackDetectNode
NODE_DISPLAY_NAME_MAPPINGS["YS_TrackDetect"] = "Track Detect (Enhanced) 🎯"

# Phase 1: Register Line Link Renderer
NODE_CLASS_MAPPINGS["YS_LineLinkRenderer"] = AdvancedLineLinkRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_LineLinkRenderer"] = "Line Link Renderer (Advanced) 🌀"

# Phase 1: Register Dot Renderer
NODE_CLASS_MAPPINGS["YS_DotRenderer"] = DotRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_DotRenderer"] = "Dot Renderer ⚫"

# Phase 1: Register Palette Map
NODE_CLASS_MAPPINGS["YS_PaletteMap"] = PaletteMapNode
NODE_DISPLAY_NAME_MAPPINGS["YS_PaletteMap"] = "Palette Map 🎨"

# Phase 1: Register Layer Merge
NODE_CLASS_MAPPINGS["YS_LayerMerge"] = LayerMergeNode
NODE_DISPLAY_NAME_MAPPINGS["YS_LayerMerge"] = "Layer Merge 🔀"

# Phase 1: Register Composite Over
NODE_CLASS_MAPPINGS["YS_CompositeOver"] = CompositeOverNode
NODE_DISPLAY_NAME_MAPPINGS["YS_CompositeOver"] = "Composite Over 🎬"

# Phase 1.5: Register Image Size Detector
NODE_CLASS_MAPPINGS["YS_ImageSizeDetector"] = ImageSizeDetectorNode
NODE_DISPLAY_NAME_MAPPINGS["YS_ImageSizeDetector"] = "Image Size Detector 📐"

# Phase 1.5: Register Video Frame Offset
NODE_CLASS_MAPPINGS["YS_VideoFrameOffset"] = VideoFrameOffsetNode
NODE_DISPLAY_NAME_MAPPINGS["YS_VideoFrameOffset"] = "Video Frame Offset 🎬"

# Phase 2: Register Bounding Box Renderer
NODE_CLASS_MAPPINGS["YS_BBoxRenderer"] = BoundingBoxRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_BBoxRenderer"] = "Bounding Box Renderer 📦"

# Phase 2: Register Blur Region Renderer
NODE_CLASS_MAPPINGS["YS_BlurRegionRenderer"] = BlurRegionRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_BlurRegionRenderer"] = "Blur Region Renderer 🌫️"

# Phase 2: Register Machine Vision Look Renderer
NODE_CLASS_MAPPINGS["YS_MVLookRenderer"] = MVLookRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_MVLookRenderer"] = "Machine Vision Look 📹"

# Simplified Line Renderers: Register Curved Line Renderer
NODE_CLASS_MAPPINGS["YS_CurvedLineRenderer"] = CurvedLineRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_CurvedLineRenderer"] = "Curved Line Renderer 〰️"

# Simplified Line Renderers: Register Graph Line Renderer
NODE_CLASS_MAPPINGS["YS_GraphLineRenderer"] = GraphLineRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_GraphLineRenderer"] = "Graph Line Renderer 🕸️"

# Simplified Line Renderers: Register Physics Line Renderer
NODE_CLASS_MAPPINGS["YS_PhysicsLineRenderer"] = PhysicsLineRendererNode
NODE_DISPLAY_NAME_MAPPINGS["YS_PhysicsLineRenderer"] = "Physics Line Renderer ⚡"

# Web UI extensions will be added here

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "__author__",
]