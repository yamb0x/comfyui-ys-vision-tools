"""
YS-vision-tools utilities package
"""

from .gpu_common import (
    GPUAccelerator,
    GPUMemoryManager,
    get_gpu_accelerator,
    is_gpu_available,
    GPU_AVAILABLE
)
from .image_utils import (
    ensure_numpy_hwc,
    ensure_torch_bchw,
    numpy_to_comfyui,
    comfyui_to_numpy,
    normalize_image,
    denormalize_image,
    create_rgba_layer,
    alpha_blend,
    resize_layer,
    rgb_to_grayscale
)
from .curve_math import (
    CurveGenerator,
    GraphBuilder,
    GPUCurveBatchGenerator,
    CUPY_AVAILABLE
)
from .optical_flow import (
    backward_warp,
    backward_warp_gpu,
    backward_warp_cpu,
    estimate_optical_flow_gpu
)
from .sdf_font import (
    SDFFontAtlas,
    get_font_atlas
)
from .color_utils import (
    normalize_color_to_rgba01,
    parse_legacy_color_string
)

__all__ = [
    # GPU utilities
    'GPUAccelerator',
    'GPUMemoryManager',
    'get_gpu_accelerator',
    'is_gpu_available',
    'GPU_AVAILABLE',

    # Image utilities
    'ensure_numpy_hwc',
    'ensure_torch_bchw',
    'numpy_to_comfyui',
    'comfyui_to_numpy',
    'normalize_image',
    'denormalize_image',
    'create_rgba_layer',
    'alpha_blend',
    'resize_layer',
    'rgb_to_grayscale',

    # Curve math
    'CurveGenerator',
    'GraphBuilder',
    'GPUCurveBatchGenerator',
    'CUPY_AVAILABLE',

    # Optical flow
    'backward_warp',
    'backward_warp_gpu',
    'backward_warp_cpu',
    'estimate_optical_flow_gpu',

    # SDF font rendering
    'SDFFontAtlas',
    'get_font_atlas',

    # Color utilities
    'normalize_color_to_rgba01',
    'parse_legacy_color_string',
]