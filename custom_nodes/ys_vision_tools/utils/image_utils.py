"""
Image format conversion utilities for YS-vision-tools
Handles conversions between ComfyUI, PyTorch, OpenCV, and NumPy formats
"""

import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

# PyTorch is always available in ComfyUI environment
try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = None


def ensure_numpy_hwc(image: Union[np.ndarray, 'TorchTensor']) -> np.ndarray:
    """
    Ensure image is in NumPy HWC (Height, Width, Channels) format

    ComfyUI uses PyTorch tensors in BHWC format (Batch, Height, Width, Channels)
    OpenCV uses NumPy arrays in HWC format (Height, Width, Channels)

    Args:
        image: Input image in various formats

    Returns:
        NumPy array in HWC format with shape (H, W, C) or (H, W) for grayscale
    """
    # Convert PyTorch tensor to NumPy
    if torch is not None and torch.is_tensor(image):
        image = image.cpu().numpy()

    # Handle different shapes
    if len(image.shape) == 4:  # BCHW format (batch)
        # Take first image from batch
        image = image[0]

    if len(image.shape) == 3:
        # Check if CHW or HWC
        if image.shape[0] in [1, 3, 4]:  # Likely CHW (channels first)
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC
        # else: already HWC

    # Squeeze single channel
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    return image


def ensure_torch_bchw(image: Union[np.ndarray, 'TorchTensor']) -> 'TorchTensor':
    """
    Ensure image is in PyTorch BCHW (Batch, Channels, Height, Width) format

    Args:
        image: Input image in various formats

    Returns:
        PyTorch tensor in BCHW format with shape (B, C, H, W)
    """
    if torch is None:
        raise ImportError("PyTorch is required for this function")

    # Convert to NumPy first if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    # Ensure HWC format
    image = ensure_numpy_hwc(image)

    # Add channel dimension if grayscale
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    # Convert HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]

    # Convert to PyTorch tensor
    return torch.from_numpy(image)


def numpy_to_comfyui(image: np.ndarray) -> 'TorchTensor':
    """
    Convert NumPy image to ComfyUI format (BHWC tensor)

    IMPORTANT: ComfyUI uses (Batch, Height, Width, Channels) format,
    NOT the standard PyTorch (Batch, Channels, Height, Width) format!

    Args:
        image: NumPy array in HWC format

    Returns:
        PyTorch tensor in BHWC format (ComfyUI standard)
    """
    if torch is None:
        raise ImportError("PyTorch is required for ComfyUI nodes")

    # Ensure HWC format first
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = ensure_numpy_hwc(image)

    # Add channel dimension if grayscale
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    # Add batch dimension if needed (keep HWC, don't transpose!)
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]  # Now (1, H, W, C)

    # Convert to PyTorch tensor (keep BHWC format for ComfyUI!)
    return torch.from_numpy(image.astype(np.float32))


def comfyui_to_numpy(image: 'TorchTensor') -> np.ndarray:
    """
    Convert ComfyUI image to NumPy format (HWC)

    Args:
        image: PyTorch tensor in BHWC format (ComfyUI standard)

    Returns:
        NumPy array in HWC format
    """
    return ensure_numpy_hwc(image)


def normalize_image(image: np.ndarray, input_range: str = 'auto') -> np.ndarray:
    """
    Normalize image to [0, 1] float32 range

    Args:
        image: Input image array
        input_range: 'auto', '0-255', '0-1', or 'custom'

    Returns:
        Normalized image in [0, 1] range
    """
    image = image.astype(np.float32)

    if input_range == 'auto':
        if image.max() > 1.0:
            input_range = '0-255'
        else:
            input_range = '0-1'

    if input_range == '0-255':
        image = image / 255.0
    elif input_range == 'custom':
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)

    # Ensure [0, 1] range
    return np.clip(image, 0.0, 1.0)


def denormalize_image(image: np.ndarray, output_range: str = '0-255') -> np.ndarray:
    """
    Denormalize image from [0, 1] to target range

    Args:
        image: Normalized image in [0, 1] range
        output_range: '0-255' or '0-1'

    Returns:
        Denormalized image
    """
    image = np.clip(image, 0.0, 1.0)

    if output_range == '0-255':
        return (image * 255).astype(np.uint8)
    else:
        return image.astype(np.float32)


def create_rgba_layer(height: int, width: int, dtype: type = np.float32) -> np.ndarray:
    """
    Create empty RGBA layer

    Args:
        height: Layer height in pixels
        width: Layer width in pixels
        dtype: Data type (default: np.float32)

    Returns:
        Zero-initialized RGBA array with shape (H, W, 4)
    """
    return np.zeros((height, width, 4), dtype=dtype)


def alpha_blend(foreground: np.ndarray, background: np.ndarray,
                alpha: Union[float, np.ndarray] = None) -> np.ndarray:
    """
    Alpha blend foreground over background

    Args:
        foreground: Foreground image (H, W, C) with or without alpha
        background: Background image (H, W, C) with or without alpha
        alpha: Optional alpha mask or scalar (default: use foreground alpha if available)

    Returns:
        Blended image

    Note:
        If dimensions don't match, foreground will be resized to match background
    """
    # Resize foreground to match background if dimensions differ
    if foreground.shape[:2] != background.shape[:2]:
        import cv2
        target_height, target_width = background.shape[:2]
        foreground = cv2.resize(foreground, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Ensure both have alpha channel
    if foreground.shape[-1] == 3:
        if alpha is None:
            alpha = np.ones((*foreground.shape[:2], 1), dtype=foreground.dtype)
        elif isinstance(alpha, (int, float)):
            alpha = np.full((*foreground.shape[:2], 1), alpha, dtype=foreground.dtype)
        elif len(alpha.shape) == 2:
            alpha = alpha[:, :, np.newaxis]
        foreground = np.concatenate([foreground, alpha], axis=-1)

    if background.shape[-1] == 3:
        background = np.concatenate([
            background,
            np.ones((*background.shape[:2], 1), dtype=background.dtype)
        ], axis=-1)

    # Extract alpha channels
    fg_alpha = foreground[:, :, 3:4]
    bg_alpha = background[:, :, 3:4]

    # Compute output alpha
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)

    # Avoid division by zero
    out_alpha_safe = np.where(out_alpha > 0, out_alpha, 1)

    # Blend RGB channels
    fg_rgb = foreground[:, :, :3]
    bg_rgb = background[:, :, :3]

    out_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)) / out_alpha_safe

    # Combine RGB and alpha
    result = np.concatenate([out_rgb, out_alpha], axis=-1)

    return result


def resize_layer(layer: np.ndarray, target_size: Tuple[int, int],
                 interpolation: str = 'linear') -> np.ndarray:
    """
    Resize layer with proper interpolation

    Args:
        layer: Input layer (H, W, C)
        target_size: (width, height) tuple
        interpolation: 'nearest', 'linear', 'cubic', or 'area'

    Returns:
        Resized layer
    """
    import cv2

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }

    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    return cv2.resize(layer, target_size, interpolation=interp_flag)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using standard weights

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Grayscale image (H, W)
    """
    if len(image.shape) == 2:
        return image

    if image.shape[-1] == 4:  # RGBA
        image = image[:, :, :3]

    # Standard RGB to grayscale weights
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image, weights)
