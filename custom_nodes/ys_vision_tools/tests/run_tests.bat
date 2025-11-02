@echo off
REM Test runner script for ys_vision_tools

echo ================================================
echo YS-vision-tools Test Suite
echo ================================================
echo.

REM Check if opencv-python is installed
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo [WARN] opencv-python not found, installing...
    pip install opencv-python
    echo.
)

echo Running GPU BBox Renderer Tests...
echo ================================================
pytest test_gpu_bbox_renderer.py -v -s
echo.

echo Running Visual Regression Tests...
echo ================================================
pytest test_gpu_visual_regression.py -v -s -m visual
echo.

echo ================================================
echo Test suite complete!
echo ================================================
pause
