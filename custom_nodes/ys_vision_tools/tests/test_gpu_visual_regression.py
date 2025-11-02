"""
Visual Regression Tests for GPU Bbox Renderer

Saves visual comparison images to verify GPU output quality
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

# Output directory for visual comparisons
OUTPUT_DIR = Path(__file__).parent / "visual_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_comparison_image(cpu_array, gpu_array, diff_array, filename):
    """
    Save side-by-side comparison image

    Args:
        cpu_array: CPU output (H, W, 4)
        gpu_array: GPU output (H, W, 4)
        diff_array: Absolute difference (H, W, 4)
        filename: Output filename
    """
    try:
        import cv2

        # Convert RGBA to BGR for saving
        cpu_bgr = cv2.cvtColor((cpu_array[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        gpu_bgr = cv2.cvtColor((gpu_array[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        diff_bgr = cv2.cvtColor((diff_array[:, :, :3] * 255 * 10).astype(np.uint8), cv2.COLOR_RGB2BGR)  # 10× for visibility

        # Create side-by-side comparison
        h, w = cpu_bgr.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = cpu_bgr
        comparison[:, w:w*2] = gpu_bgr
        comparison[:, w*2:] = diff_bgr

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "CPU", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "GPU", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Diff (10×)", (w*2 + 10, 30), font, 1, (255, 255, 255), 2)

        # Save
        output_path = OUTPUT_DIR / filename
        cv2.imwrite(str(output_path), comparison)
        print(f"Saved visual comparison: {output_path}")

    except ImportError:
        print("OpenCV not available, skipping visual comparison save")


class TestGPUVisualRegression:
    """Visual regression tests with saved outputs"""

    def setup_method(self):
        """Setup test fixtures"""
        self.node = BoundingBoxRendererNode()
        self.width = 1920
        self.height = 1080

    @pytest.mark.visual
    def test_single_box_visual(self):
        """Visual test: single box"""
        tracks = np.array([[960.0, 540.0]])

        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=3.0,
            fill_opacity=0.5,
            roundness=0.3,
            tracks=tracks,
            width=100,
            height=100,
            color="1.0,0.5,0.0",  # Orange
            use_gpu=True
        )

        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=3.0,
            fill_opacity=0.5,
            roundness=0.3,
            tracks=tracks,
            width=100,
            height=100,
            color="1.0,0.5,0.0",
            use_gpu=False
        )

        gpu_array = gpu_output.cpu().numpy().squeeze()
        cpu_array = cpu_output.cpu().numpy().squeeze()
        diff_array = np.abs(gpu_array - cpu_array)

        save_comparison_image(cpu_array, gpu_array, diff_array, "single_box.png")

        # Assert visual quality
        max_diff = diff_array.max()
        mean_diff = diff_array.mean()

        print(f"Visual metrics - Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.6f}")

        assert max_diff < 0.1, f"Maximum difference too large: {max_diff}"
        assert mean_diff < 0.001, f"Mean difference too large: {mean_diff}"

    @pytest.mark.visual
    def test_grid_boxes_visual(self):
        """Visual test: grid of boxes"""
        # Create 5×5 grid
        tracks = []
        for y in range(5):
            for x in range(5):
                tracks.append([
                    self.width * (x + 1) / 6,
                    self.height * (y + 1) / 6
                ])
        tracks = np.array(tracks)

        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.5,
            tracks=tracks,
            width=80,
            height=60,
            color="0.0,0.8,1.0",  # Cyan
            use_gpu=True
        )

        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.5,
            tracks=tracks,
            width=80,
            height=60,
            color="0.0,0.8,1.0",
            use_gpu=False
        )

        gpu_array = gpu_output.cpu().numpy().squeeze()
        cpu_array = cpu_output.cpu().numpy().squeeze()
        diff_array = np.abs(gpu_array - cpu_array)

        save_comparison_image(cpu_array, gpu_array, diff_array, "grid_boxes.png")

        # SSIM-like metric
        ssim_value = 1.0 - diff_array.mean()
        print(f"Visual similarity: {ssim_value:.6f}")

        assert ssim_value > 0.99, f"Visual similarity too low: {ssim_value}"

    @pytest.mark.visual
    def test_roundness_visual(self):
        """Visual test: different roundness levels"""
        tracks = np.array([
            [400.0, 540.0],   # Square
            [960.0, 540.0],   # Semi-rounded
            [1520.0, 540.0],  # Circular
        ])

        for roundness, name in [(0.0, "square"), (0.5, "semi"), (1.0, "circle")]:
            (gpu_output,) = self.node.execute(
                image_width=self.width,
                image_height=self.height,
                box_mode="fixed",
                stroke_px=3.0,
                fill_opacity=0.4,
                roundness=roundness,
                tracks=np.array([[960.0, 540.0]]),
                width=120,
                height=120,
                color="1.0,0.0,1.0",  # Magenta
                use_gpu=True
            )

            (cpu_output,) = self.node.execute(
                image_width=self.width,
                image_height=self.height,
                box_mode="fixed",
                stroke_px=3.0,
                fill_opacity=0.4,
                roundness=roundness,
                tracks=np.array([[960.0, 540.0]]),
                width=120,
                height=120,
                color="1.0,0.0,1.0",
                use_gpu=False
            )

            gpu_array = gpu_output.cpu().numpy().squeeze()
            cpu_array = cpu_output.cpu().numpy().squeeze()
            diff_array = np.abs(gpu_array - cpu_array)

            save_comparison_image(cpu_array, gpu_array, diff_array, f"roundness_{name}.png")

            # Check edge quality (important for AA)
            edge_diff = diff_array[diff_array > 0].mean() if (diff_array > 0).any() else 0
            print(f"Roundness {roundness} - Edge diff: {edge_diff:.6f}")

            assert edge_diff < 0.01, f"Edge quality degraded at roundness={roundness}"

    @pytest.mark.visual
    def test_stress_many_boxes_visual(self):
        """Visual test: stress test with 200 boxes"""
        np.random.seed(42)
        tracks = np.random.rand(200, 2) * [[self.width, self.height]]

        (gpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=1.5,
            fill_opacity=0.2,
            roundness=0.3,
            tracks=tracks,
            width=40,
            height=40,
            color="0.2,1.0,0.2",  # Lime green
            use_gpu=True
        )

        (cpu_output,) = self.node.execute(
            image_width=self.width,
            image_height=self.height,
            box_mode="fixed",
            stroke_px=1.5,
            fill_opacity=0.2,
            roundness=0.3,
            tracks=tracks,
            width=40,
            height=40,
            color="0.2,1.0,0.2",
            use_gpu=False
        )

        gpu_array = gpu_output.cpu().numpy().squeeze()
        cpu_array = cpu_output.cpu().numpy().squeeze()
        diff_array = np.abs(gpu_array - cpu_array)

        save_comparison_image(cpu_array, gpu_array, diff_array, "stress_200_boxes.png")

        # Check that cumulative error doesn't build up
        max_diff = diff_array.max()
        mean_diff = diff_array.mean()

        print(f"Stress test (200 boxes) - Max: {max_diff:.4f}, Mean: {mean_diff:.6f}")

        assert max_diff < 0.15, f"Cumulative error too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean error too large: {mean_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "visual"])
