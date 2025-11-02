# YS-Vision v2 Development Plan - PROJECT OVERVIEW

## 🎯 What You're Building

You are building **YS-Vision v2**, a ComfyUI custom node pack for creating multi-color layered vision overlays on video frames. Think of it as a visual effects toolkit that:

1. **Tracks points** in video frames (like feature detection in computer vision)
2. **Renders overlays** (dots, lines, bounding boxes, blur regions, HUD text)
3. **Composites layers** with different blend modes onto original footage

## 🧩 The Big Picture

```
Video Frames → Track Detection → Multiple Renderers → Layer Merging → Final Composite
```

### Core Concepts You Need to Know

**ComfyUI**: A node-based interface for AI workflows. Your nodes will plug into this system.

**Tracking**: Finding and following points/features across video frames.

**Rendering**: Drawing visual elements (dots, lines, boxes) based on tracked points.

**Layer Compositing**: Combining multiple transparent overlays using blend modes (like Photoshop layers).

## 📦 What's Already Done

- **Documentation**: Comprehensive specs in `/docs/` (00-90)
- **Architecture**: Fully designed node interfaces and data contracts
- **Standards**: Quality playbook and testing approach defined

## 🛠 Tech Stack You'll Use

- **Python 3.10+**: Primary language
- **ComfyUI**: The framework your nodes integrate with
- **OpenCV**: Computer vision operations (`pip install opencv-python`)
- **NumPy**: Array operations (`pip install numpy`)
- **PyTorch**: Tensor operations (required by ComfyUI)
- **SciPy**: Scientific computing for advanced features

## 🎓 Domain Knowledge Primer

### Computer Vision Basics
- **Feature Detection**: Finding interesting points in images (corners, blobs)
- **Tracking**: Following points across frames (KLT tracker)
- **Thresholding**: Converting grayscale to binary based on intensity
- **Blur/Smoothing**: Reducing noise in images

### Graphics Concepts
- **RGBA**: Red, Green, Blue, Alpha (transparency) channels
- **Premultiplied Alpha**: RGB values pre-multiplied by alpha for correct blending
- **Blend Modes**: How layers combine (add, screen, multiply, etc.)
- **Anti-aliasing (AA)**: Smoothing jagged edges by supersampling

### ComfyUI Specifics
- **Nodes**: Processing units with inputs/outputs
- **IMAGE Format**: Tensors of shape (batch, height, width, channels)
- **Execution**: Nodes run when their inputs are ready

## 📋 Development Phases

### Phase 1: MVP (Core Tracking & Basic Rendering)
Essential nodes to get a working system

### Phase 2: Extended Features
Additional renderers and effects

### Phase 3: Optimization
GPU acceleration and performance

### Phase 4: Advanced
Clustering, complex curves, region-aware processing

## ⚠️ Critical Success Factors

1. **Test Everything**: Write tests BEFORE implementation (TDD)
2. **Small Commits**: Commit after each working feature
3. **DRY**: Don't repeat code - extract common functions
4. **YAGNI**: Only build what's specified, no extras
5. **Documentation**: Comment complex algorithms

## 🔍 Where to Find Help

- `/docs/00-Overview.md`: Project vision
- `/docs/20-Architecture.md`: System design
- `/docs/30-Nodes.md`: Node specifications
- `/docs/50-QualityPlaybook.md`: Quality standards
- ComfyUI Discord/Forums: Community help
- OpenCV Documentation: Vision algorithms

## Next Steps

Continue to `01-ENVIRONMENT-SETUP.md` for development environment setup.