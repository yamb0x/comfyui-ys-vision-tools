"""
GPU acceleration utilities for YS-vision-tools
Provides GPU-accelerated operations using CuPy for NVIDIA RTX 5090
"""

import numpy as np
from typing import Union, Optional, Tuple, TYPE_CHECKING
import warnings

# Try to import GPU libraries
try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        # Set up memory management for RTX 5090 (24GB VRAM)
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=8 * 1024**3)  # Use 8GB max per operation

        # PyTorch GPU settings
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    torch = None
    warnings.warn("GPU libraries not available. Install cupy-cuda12x and torch for GPU acceleration.")

# Type hints that work even when cp is None
if TYPE_CHECKING:
    from cupy import ndarray as CupyArray
else:
    CupyArray = None


class GPUAccelerator:
    """Manage GPU acceleration for YS-vision-tools on RTX 5090"""

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU accelerator

        Args:
            device_id: CUDA device ID (default: 0)
        """
        self.use_gpu = GPU_AVAILABLE
        self.device_id = device_id

        if self.use_gpu:
            self.device = torch.device(f'cuda:{device_id}')
            cp.cuda.Device(device_id).use()
        else:
            self.device = torch.device('cpu')
            warnings.warn("GPU not available, falling back to CPU")

    def to_gpu(self, array: np.ndarray) -> Union['CupyArray', np.ndarray]:
        """
        Transfer numpy array to GPU memory

        Args:
            array: NumPy array to transfer

        Returns:
            CuPy array if GPU available, otherwise original numpy array
        """
        if self.use_gpu and cp is not None:
            return cp.asarray(array)
        return array

    def to_cpu(self, array: Union['CupyArray', np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """
        Transfer array back to CPU memory

        Args:
            array: CuPy array, PyTorch tensor, or NumPy array

        Returns:
            NumPy array
        """
        if isinstance(array, np.ndarray):
            return array
        elif self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        elif torch.is_tensor(array):
            return array.cpu().numpy()
        return array

    def gpu_convolve2d(self, image: np.ndarray, kernel: np.ndarray,
                       mode: str = 'same') -> np.ndarray:
        """
        GPU-accelerated 2D convolution

        Args:
            image: Input image array
            kernel: Convolution kernel
            mode: Convolution mode ('same', 'valid', 'full')

        Returns:
            Convolved image as numpy array
        """
        if self.use_gpu and cp is not None:
            img_gpu = cp.asarray(image)
            kernel_gpu = cp.asarray(kernel)

            # Use cupyx for signal processing
            from cupyx.scipy import signal as cp_signal
            result = cp_signal.convolve2d(img_gpu, kernel_gpu, mode=mode)

            return cp.asnumpy(result)
        else:
            # CPU fallback
            from scipy import signal
            return signal.convolve2d(image, kernel, mode=mode)

    def gpu_fft2(self, image: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated 2D FFT

        Args:
            image: Input image array

        Returns:
            FFT result as numpy array
        """
        if self.use_gpu and cp is not None:
            img_gpu = cp.asarray(image)
            fft_result = cp.fft.fft2(img_gpu)
            return cp.asnumpy(fft_result)
        else:
            return np.fft.fft2(image)

    def gpu_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated gradient computation

        Args:
            image: Input image array

        Returns:
            Tuple of (gradient_x, gradient_y) as numpy arrays
        """
        if self.use_gpu and cp is not None:
            img_gpu = cp.asarray(image)
            grad_y, grad_x = cp.gradient(img_gpu)
            return cp.asnumpy(grad_x), cp.asnumpy(grad_y)
        else:
            grad_y, grad_x = np.gradient(image)
            return grad_x, grad_y

    def synchronize(self):
        """Synchronize GPU operations"""
        if self.use_gpu and cp is not None:
            cp.cuda.Stream.null.synchronize()

    def memory_stats(self) -> dict:
        """
        Get GPU memory statistics

        Returns:
            Dictionary with memory usage information
        """
        if not self.use_gpu or cp is None:
            return {"gpu_available": False}

        mempool = cp.get_default_memory_pool()

        return {
            "gpu_available": True,
            "used_bytes": mempool.used_bytes(),
            "used_mb": mempool.used_bytes() / 1024**2,
            "used_gb": mempool.used_bytes() / 1024**3,
            "total_bytes": mempool.total_bytes(),
            "total_mb": mempool.total_bytes() / 1024**2,
            "total_gb": mempool.total_bytes() / 1024**3,
        }

    def clear_memory(self):
        """Clear GPU memory pool"""
        if self.use_gpu and cp is not None:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


class GPUMemoryManager:
    """Efficient GPU memory management for RTX 5090"""

    def __init__(self, gpu_memory_gb: int = 24):
        """
        Initialize memory manager

        Args:
            gpu_memory_gb: Total GPU memory in GB (default: 24 for RTX 5090)
        """
        self.total_memory = gpu_memory_gb * 1024**3
        self.reserved = int(self.total_memory * 0.8)  # Reserve 80%

        if GPU_AVAILABLE and cp is not None:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=self.reserved)

    def profile_operation(self, func, *args, **kwargs):
        """
        Profile GPU memory and time for an operation

        Args:
            func: Function to profile
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, elapsed_ms, memory_used_mb)
        """
        if not GPU_AVAILABLE or cp is None:
            import time
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            return result, elapsed_ms, 0

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
        memory_used_mb = (used_after - used_before) / 1024**2

        return result, elapsed_ms, memory_used_mb


# Global GPU accelerator instance
_gpu_accelerator = None


def get_gpu_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available"""
    return GPU_AVAILABLE
