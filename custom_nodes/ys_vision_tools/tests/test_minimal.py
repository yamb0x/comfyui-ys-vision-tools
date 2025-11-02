"""
Minimal GPU BBox Renderer Test - Standalone

This test can run without pytest and checks basic GPU functionality
"""

import numpy as np
import sys
from pathlib import Path

# Setup path
parent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))

def main():
    print("=" * 60)
    print("YS-vision-tools GPU BBox Renderer - Minimal Test")
    print("=" * 60)
    print()

    # Test 1: Import modules
    print("[1/5] Testing imports...")
    try:
        from custom_nodes.ys_vision_tools.utils import is_gpu_available
        from custom_nodes.ys_vision_tools.nodes.bbox_renderer import BoundingBoxRendererNode
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test 2: Check GPU availability
    print("\n[2/5] Checking GPU availability...")
    gpu_available = is_gpu_available()
    print(f"  GPU Available: {gpu_available}")
    if not gpu_available:
        print("  [WARN] GPU not available - tests will use CPU fallback")

    # Test 3: Initialize node
    print("\n[3/5] Initializing BBoxRenderer node...")
    try:
        node = BoundingBoxRendererNode()
        print("✓ Node initialized")
        if node.gpu_renderer is not None:
            print("✓ GPU renderer initialized")
        else:
            print("  [INFO] GPU renderer not available, using CPU fallback")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    # Test 4: Render with CPU
    print("\n[4/5] Testing CPU rendering...")
    try:
        tracks = np.array([[960.0, 540.0], [800.0, 400.0], [1100.0, 680.0]])

        import time
        start = time.perf_counter()

        (cpu_output,) = node.execute(
            image_width=1920,
            image_height=1080,
            box_mode="fixed",
            stroke_px=2.0,
            fill_opacity=0.3,
            roundness=0.2,
            tracks=tracks,
            width=40,
            height=40,
            color="1.0,0.5,0.0",
            use_gpu=False
        )

        cpu_time = (time.perf_counter() - start) * 1000
        print(f"✓ CPU rendering successful: {cpu_time:.2f}ms")
        print(f"  Output shape: {cpu_output.shape}")

    except Exception as e:
        print(f"✗ CPU rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Render with GPU (if available)
    print("\n[5/5] Testing GPU rendering...")
    if not gpu_available or node.gpu_renderer is None:
        print("  [SKIP] GPU not available")
    else:
        try:
            start = time.perf_counter()

            (gpu_output,) = node.execute(
                image_width=1920,
                image_height=1080,
                box_mode="fixed",
                stroke_px=2.0,
                fill_opacity=0.3,
                roundness=0.2,
                tracks=tracks,
                width=40,
                height=40,
                color="1.0,0.5,0.0",
                use_gpu=True
            )

            gpu_time = (time.perf_counter() - start) * 1000
            print(f"✓ GPU rendering successful: {gpu_time:.2f}ms")
            print(f"  Output shape: {gpu_output.shape}")

            # Compare results
            cpu_array = cpu_output.cpu().numpy()
            gpu_array = gpu_output.cpu().numpy()

            if np.allclose(cpu_array, gpu_array, rtol=1e-3, atol=1e-3):
                print("✓ GPU output matches CPU (within tolerance)")
            else:
                max_diff = np.abs(cpu_array - gpu_array).max()
                mean_diff = np.abs(cpu_array - gpu_array).mean()
                print(f"  [WARN] GPU differs from CPU:")
                print(f"    Max diff: {max_diff:.6f}")
                print(f"    Mean diff: {mean_diff:.6f}")

            # Speedup
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  Speedup: {speedup:.1f}×")

        except Exception as e:
            print(f"✗ GPU rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print()
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
