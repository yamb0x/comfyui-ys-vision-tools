"""
Color Utility Functions for YS-vision-tools

Provides robust color parsing for ComfyUI nodes:
- Supports ComfyUI's native COLOR picker (hex strings like "#ffffff")
- Supports named colors ("red", "orange", "cyan")
- Backward compatible with float lists [r, g, b, a] or [r, g, b]
- Handles both 0-255 integer RGB and 0.0-1.0 float RGB

Usage:
    from ..utils.color_utils import normalize_color_to_rgba01

    # In node INPUT_TYPES:
    "color": ("COLOR", {"default": "#ffffff"})
    "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

    # In node execute():
    rgba = normalize_color_to_rgba01(color, alpha)
"""

from PIL import ImageColor
from typing import Union, List, Tuple


def normalize_color_to_rgba01(
    color: Union[str, List, Tuple, None],
    alpha: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Convert various color formats to normalized RGBA floats (0.0-1.0).

    Args:
        color: Color in one of these formats:
            - HEX string: "#ffcc00", "#fc0", "#ffffff"
            - Named color: "red", "orange", "cyan", "white"
            - RGB list/tuple: [255, 128, 0] or [1.0, 0.5, 0.0]
            - RGBA list/tuple: [255, 128, 0, 255] or [1.0, 0.5, 0.0, 1.0]
            - None: defaults to white
        alpha: Alpha value (0.0-1.0), only used if color doesn't include alpha

    Returns:
        Tuple of (r, g, b, a) as floats in range 0.0-1.0

    Examples:
        >>> normalize_color_to_rgba01("#ff0000", 0.8)
        (1.0, 0.0, 0.0, 0.8)

        >>> normalize_color_to_rgba01("orange")
        (1.0, 0.6470588235294118, 0.0, 1.0)

        >>> normalize_color_to_rgba01([255, 128, 0], 0.5)
        (1.0, 0.5019607843137255, 0.0, 0.5)

        >>> normalize_color_to_rgba01([1.0, 0.5, 0.0])
        (1.0, 0.5, 0.0, 1.0)
    """
    # Default to white if None
    if color is None:
        return (1.0, 1.0, 1.0, float(alpha))

    # Handle string inputs (HEX or named colors)
    if isinstance(color, str):
        # Strip whitespace and handle empty strings
        color = color.strip()
        if not color:
            return (1.0, 1.0, 1.0, float(alpha))

        try:
            # PIL ImageColor handles both hex (#ffcc00) and named colors ("orange")
            r, g, b = ImageColor.getcolor(color, "RGB")
            return (r / 255.0, g / 255.0, b / 255.0, float(alpha))
        except ValueError as e:
            print(f"[YS-COLOR] Invalid color string '{color}': {e}, defaulting to white")
            return (1.0, 1.0, 1.0, float(alpha))

    # Handle list/tuple inputs
    if isinstance(color, (list, tuple)):
        if len(color) == 4:
            # RGBA format
            r, g, b, a = color
            # Auto-detect if values are 0-255 (int RGB) or 0.0-1.0 (float RGB)
            if max(color) > 1.0:
                return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
            else:
                return (float(r), float(g), float(b), float(a))

        elif len(color) == 3:
            # RGB format (use alpha parameter)
            r, g, b = color
            # Auto-detect if values are 0-255 (int RGB) or 0.0-1.0 (float RGB)
            if max(color) > 1.0:
                return (r / 255.0, g / 255.0, b / 255.0, float(alpha))
            else:
                return (float(r), float(g), float(b), float(alpha))

        else:
            print(f"[YS-COLOR] Invalid color list length {len(color)}, expected 3 or 4, defaulting to white")
            return (1.0, 1.0, 1.0, float(alpha))

    # Unknown format
    print(f"[YS-COLOR] Unknown color format {type(color)}, defaulting to white")
    return (1.0, 1.0, 1.0, float(alpha))


def parse_legacy_color_string(color_str: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Parse legacy comma-separated RGB strings (backward compatibility).

    Args:
        color_str: String like "1.0,1.0,0.0" or "255,128,0"
        alpha: Alpha value (0.0-1.0)

    Returns:
        Tuple of (r, g, b, a) as floats in range 0.0-1.0

    Examples:
        >>> parse_legacy_color_string("1.0,1.0,0.0")
        (1.0, 1.0, 0.0, 1.0)

        >>> parse_legacy_color_string("255,128,0", 0.8)
        (1.0, 0.5019607843137255, 0.0, 0.8)
    """
    try:
        values = [float(x.strip()) for x in color_str.split(',')]
        if len(values) >= 3:
            r, g, b = values[:3]
            a = values[3] if len(values) >= 4 else alpha

            # Auto-detect int (0-255) vs float (0.0-1.0)
            if max(r, g, b) > 1.0:
                return (r / 255.0, g / 255.0, b / 255.0, a if a <= 1.0 else a / 255.0)
            else:
                return (r, g, b, a)
        else:
            print(f"[YS-COLOR] Invalid color string '{color_str}', expected 3-4 values, defaulting to white")
            return (1.0, 1.0, 1.0, float(alpha))
    except ValueError as e:
        print(f"[YS-COLOR] Failed to parse color string '{color_str}': {e}, defaulting to white")
        return (1.0, 1.0, 1.0, float(alpha))
