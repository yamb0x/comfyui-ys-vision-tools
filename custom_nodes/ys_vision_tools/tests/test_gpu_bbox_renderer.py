"""
Unit tests for GPU BBox Renderer

Tests GPU implementation against CPU reference for correctness
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add custom_nodes directory to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import from package structure
try:
    from custom_nodes.ys_vision_tools.utils import is_gpu_available
    from custom_nodes.ys_vision_tools.nodes.bbox_renderer import BoundingBoxRendererNode
except ImportError:
    # Fallback for different import contexts
    from ys_vision_tools.utils import is_gpu_available
    from ys_vision_tools.nodes.bbox_renderer import BoundingBoxRendererNode


class TestGPUBBoxRenderer:
    """Test GPU bbox renderer vs CPU reference"""

    def setup_method(self):
        """Setup test fixtures"""
        self.node = BoundingBoxRendererNode()
        self.width = 1920
        self.height = 1080

    def test_gpu_available(self):
        """Test that GPU is available (should pass on RTX 5090)"""
        assert is_gpu_available(), "GPU not available - install cupy-cuda12x"

    def test_gpu_renderer_initialized(self):
        """Test that GPU renderer was initialized"""
        if is_gpu_available():
            assert self.node.gpu_renderer is not None, "GPU renderer not initialized"

    def test_single_box_gpu_vs_cpu(self):
        """Test single box rendering: GPU vs CPU"""
        # Create test data: single box at center
        tracks = np.array([[960.0, 540.0]])  # Center of 1920x1080

        # Render with GPU
        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.5,
            roundness=0.0,
            tracks=tracks,
            width=40,
            height=40,
            color="1.0,0.0,0.0",  # Red
            use_gpu=True
        )

        # Render with CPU
        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.5,
            roundness=0.0,
            tracks=tracks,
            width=40,
            height=40,
            color="1.0,0.0,0.0",  # Red
            use_gpu=False
        )

        # Convert to numpy
        gpu_array = gpu_output.cpu().numpy()
        cpu_array = cpu_output.cpu().numpy()

        # Check shapes match
        assert gpu_array.shape == cpu_array.shape, \
            f"Shape mismatch: GPU {gpu_array.shape} vs CPU {cpu_array.shape}"

        # Check outputs are close (allow for FP precision differences)
        assert np.allclose(gpu_array, cpu_array, rtol=1e-3, atol=1e-3), \
            "GPU output differs from CPU reference"

        # Visual diff: <1% pixels should differ
        diff = np.abs(gpu_array - cpu_array)
        diff_pct = (diff > 0.01).sum() / diff.size
        assert diff_pct < 0.01, \
            f"Too many different pixels: {diff_pct*100:.2f}%"

    def test_multiple_boxes_gpu_vs_cpu(self):
        """Test multiple boxes: GPU vs CPU"""
        # Create test data: 100 random boxes
        np.random.seed(42)
        tracks = np.random.rand(100, 2) * [[self.width, self.height]]

        # Render with GPU
        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.2,
            tracks=tracks,
            width=30,
            height=30,
            color="0.0,1.0,0.0",  # Green
            use_gpu=True
        )

        # Render with CPU
        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.2,
            tracks=tracks,
            width=30,
            height=30,
            color="0.0,1.0,0.0",  # Green
            use_gpu=False
        )

        # Convert to numpy
        gpu_array = gpu_output.cpu().numpy()
        cpu_array = cpu_output.cpu().numpy()

        # Check close
        assert np.allclose(gpu_array, cpu_array, rtol=1e-3, atol=1e-3), \
            "GPU output differs from CPU for multiple boxes"

    def test_rounded_corners_gpu_vs_cpu(self):
        """Test rounded corners: GPU vs CPU"""
        tracks = np.array([[500.0, 500.0], [1000.0, 500.0], [1500.0, 500.0]])

        # Test different roundness levels
        for roundness in [0.0, 0.5, 1.0]:
            (gpu_output,) = self.node.execute(
                image_width=self.width,
                image_height=self.height,
                box_mode="fixed",
                stroke_px=3.0,
                fill_opacity=0.4,
                roundness=roundness,
                tracks=tracks,
                width=50,
                height=50,
                use_gpu=True
            )

            (cpu_output,) = self.node.execute(
                image_width=self.width,
                image_height=self.height,
                box_mode="fixed",
                stroke_px=3.0,
                fill_opacity=0.4,
                roundness=roundness,
                tracks=tracks,
                width=50,
                height=50,
                use_gpu=False
            )

            gpu_array = gpu_output.cpu().numpy()
            cpu_array = cpu_output.cpu().numpy()

            assert np.allclose(gpu_array, cpu_array, rtol=1e-3, atol=1e-3), \
                f"GPU differs from CPU at roundness={roundness}"

    def test_variable_sizes_gpu_vs_cpu(self):
        """Test variable box sizes (from_radius mode): GPU vs CPU"""
        tracks = np.array([
            [400.0, 400.0],
            [800.0, 400.0],
            [1200.0, 400.0]
        ])

        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="from_radius",
            stroke_px=2.0,
            fill_opacity=0.2,
            roundness=0.3,
            tracks=tracks,
            radius_px=30.0,
            use_gpu=True
        )

        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="from_radius",
            stroke_px=2.0,
            fill_opacity=0.2,
            roundness=0.3,
            tracks=tracks,
            radius_px=30.0,
            use_gpu=False
        )

        gpu_array = gpu_output.cpu().numpy()
        cpu_array = cpu_output.cpu().numpy()

        assert np.allclose(gpu_array, cpu_array, rtol=1e-3, atol=1e-3), \
            "GPU differs from CPU for variable sizes"

    @pytest.mark.benchmark
    def test_gpu_performance_4k(self):
        """Benchmark: 100 boxes at 4K resolution"""
        if not is_gpu_available():
            pytest.skip("GPU not available")

        # 4K resolution
        width_4k = 3840
        height_4k = 2160

        # 100 boxes
        np.random.seed(42)
        tracks = np.random.rand(100, 2) * [[width_4k, height_4k]]

        import time

        # GPU benchmark
        start = time.perf_counter()
        (gpu_output,) = self.node.execute(
            image_width=width_4k,
            image_height=height_4k,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.2,
            tracks=tracks,
            width=40,
            height=40,
            use_gpu=True
        )
        gpu_time = (time.perf_counter() - start) * 1000

        # CPU benchmark
        start = time.perf_counter()
        (cpu_output,) = self.node.execute(
            image_width=width_4k,
            image_height=height_4k,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.2,
            tracks=tracks,
            width=40,
            height=40,
            use_gpu=False
        )
        cpu_time = (time.perf_counter() - start) * 1000

        speedup = cpu_time / gpu_time

        print(f"\n4K Benchmark (100 boxes):")
        print(f"  CPU: {cpu_time:.2f}ms")
        print(f"  GPU: {gpu_time:.2f}ms")
        print(f"  Speedup: {speedup:.1f}×")

        # Assert performance targets
        assert gpu_time < 10, f"GPU too slow: {gpu_time:.2f}ms (target: <10ms)"
        assert speedup > 10, f"Speedup too low: {speedup:.1f}× (target: >10×)"

    def test_batch_mode_gpu(self):
        """Test batch mode (animation frames) with GPU"""
        # 5 frames of animation
        frames_tracks = []
        for frame in range(5):
            np.random.seed(frame)
            tracks = np.random.rand(20, 2) * [[self.width, self.height]]
            frames_tracks.append(tracks)

        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.1,
            tracks=frames_tracks,  # List of arrays
            width=25,
            height=25,
            use_gpu=True
        )

        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.1,
            tracks=frames_tracks,
            width=25,
            height=25,
            use_gpu=False
        )

        gpu_array = gpu_output.cpu().numpy()
        cpu_array = cpu_output.cpu().numpy()

        # Should be (5, H, W, 4) - batch dimension
        assert gpu_array.shape[0] == 5, f"Expected 5 frames, got {gpu_array.shape[0]}"
        assert np.allclose(gpu_array, cpu_array, rtol=1e-3, atol=1e-3), \
            "GPU batch differs from CPU batch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
