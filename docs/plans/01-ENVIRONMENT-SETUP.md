# Environment Setup Guide

## 📍 Prerequisites Check

Before starting, verify you have:
- [ ] Python 3.10 or higher (`python --version`)
- [ ] Git installed (`git --version`)
- [ ] ComfyUI installed and running
- [ ] A code editor (VS Code recommended)
- [ ] Terminal/Command Prompt access

## 🔧 Step-by-Step Setup

### 1. Clone and Navigate to Project
```bash
cd path/to/comfyui/custom_nodes
git clone [your-repo-url] ys_vision
cd ys_vision
```

### 2. Create Python Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Core Dependencies
```bash
# Essential packages for Phase 1
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install Pillow==10.1.0

# ComfyUI should provide torch, but verify:
python -c "import torch; print(torch.__version__)"
```

### 4. Create Project Structure
```bash
# Run this from project root (ys_vision/)
mkdir -p custom_nodes/ys_vision/nodes
mkdir -p custom_nodes/ys_vision/assets/fonts
mkdir -p custom_nodes/ys_vision/assets/styles
mkdir -p tests
mkdir -p tests/fixtures
mkdir -p tests/unit
mkdir -p tests/integration
```

### 5. Initialize Python Package
Create `custom_nodes/ys_vision/__init__.py`:
```python
"""YS-Vision v2: Multi-Color Layered Vision Overlays for ComfyUI"""
__version__ = "2.0.0"

# Node registration will go here
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
```

### 6. Set Up Testing Framework
```bash
pip install pytest==7.4.3
pip install pytest-cov==4.1.0
pip install pytest-mock==3.12.0

# Create pytest configuration
echo "[tool.pytest.ini_options]
testpaths = [\"tests\"]
python_files = \"test_*.py\"
python_classes = \"Test*\"
python_functions = \"test_*\"
addopts = \"-v --cov=custom_nodes/ys_vision --cov-report=term-missing\"
" > pytest.ini
```

### 7. Create Development Scripts
Create `scripts/test.sh` (Linux/Mac) or `scripts/test.bat` (Windows):

**test.sh:**
```bash
#!/bin/bash
pytest tests/ -v --cov=custom_nodes/ys_vision --cov-report=html
```

**test.bat:**
```batch
@echo off
pytest tests/ -v --cov=custom_nodes/ys_vision --cov-report=html
```

### 8. Verify ComfyUI Integration
Create a test node to verify setup:

`custom_nodes/ys_vision/nodes/test_node.py`:
```python
class TestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello YS-Vision!"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "YS-Vision/Test"

    def execute(self, text):
        return (f"Test: {text}",)
```

Update `custom_nodes/ys_vision/__init__.py`:
```python
from .nodes.test_node import TestNode

NODE_CLASS_MAPPINGS = {
    "YSVisionTest": TestNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YSVisionTest": "YS Vision Test Node"
}
```

### 9. Test the Setup
1. Restart ComfyUI
2. Check if "YS Vision Test Node" appears in the node menu
3. Run basic tests: `pytest tests/`

## 🔍 Troubleshooting

### ComfyUI doesn't see the nodes
- Check you're in the right directory: `ComfyUI/custom_nodes/ys_vision/`
- Verify `__init__.py` has NODE_CLASS_MAPPINGS
- Check ComfyUI console for import errors

### Import errors
- Ensure virtual environment is activated
- Verify all packages installed: `pip list`
- Check Python version: `python --version`

### OpenCV issues
- On Linux: `sudo apt-get install python3-opencv`
- On Mac: `brew install opencv`
- On Windows: May need Visual C++ redistributables

## 📝 Environment Checklist

Before proceeding to Phase 1:
- [ ] Python environment set up
- [ ] All dependencies installed
- [ ] Project structure created
- [ ] Test framework working
- [ ] ComfyUI recognizes test node
- [ ] Can run `pytest` successfully

## Next Steps
Continue to `02-PHASE1-MVP.md` for Phase 1 implementation tasks.