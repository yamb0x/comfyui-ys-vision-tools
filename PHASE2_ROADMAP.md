# Phase 2 Roadmap - Extended Renderers

**Start Date:** November 2, 2025
**Status:** Planning / Ready to Begin
**Prerequisites:** ✅ Phase 1 Complete & Deployed

---

## 🎯 Phase 2 Goals

Add 4 powerful renderer nodes that extend the visualization capabilities of YS-vision-tools:

1. **BoundingBoxRenderer** - Draw boxes around tracked objects
2. **BlurRegionRenderer** - Apply selective blur effects
3. **HUDTextRenderer** - Overlay technical data and metadata
4. **MVLookRenderer** - Apply color grading and LUTs

---

## 📦 Phase 2 Deliverables

### 1. BoundingBoxRenderer Node 📦

**Purpose:** Draw customizable bounding boxes around tracked points or detection regions

**Key Features:**
- Multiple box sizing modes: fixed, from radius, from age
- Configurable stroke and fill
- Rounded corners support
- Color palette integration
- GPU-accelerated rendering

**Inputs:**
- Tracks or detection boxes
- Box dimensions/modes
- Styling options (stroke, fill, roundness)
- Colors/palettes

**Outputs:**
- RGBA layer with rendered boxes

**Use Cases:**
- Object highlighting in detection workflows
- Age-based visualization (older tracks = larger boxes)
- Region-of-interest marking
- Technical overlays for analysis

---

### 2. BlurRegionRenderer Node 🌫️

**Purpose:** Apply selective blur effects based on tracked regions

**Key Features:**
- Gaussian, motion, and radial blur modes
- Region-based masking from tracks
- Feathering/transition control
- Blur strength gradients
- GPU-accelerated blur kernels

**Inputs:**
- Base image
- Tracks or mask regions
- Blur parameters (type, strength, feather)
- Region sizing options

**Outputs:**
- IMAGE with selective blur applied

**Use Cases:**
- Privacy protection (blur faces/objects)
- Depth-of-field simulation
- Focus effects (blur everything except tracked region)
- Motion blur on moving objects
- Artistic effects

---

### 3. HUDTextRenderer Node 📊

**Purpose:** Overlay technical data, metadata, and UI elements

**Key Features:**
- Dynamic text from track data (position, velocity, confidence)
- Multiple font rendering options
- Customizable layouts (corner placement, tracking labels)
- Real-time data display
- Anti-aliased text rendering

**Inputs:**
- Tracks with metadata
- Text formatting options
- Position/layout configuration
- Style settings (font, size, color, shadow)

**Outputs:**
- RGBA layer with rendered text
- Optional: direct IMAGE output

**Use Cases:**
- Technical analysis overlays
- Track ID and confidence display
- Frame counters and timestamps
- Debugging visualization
- Production HUD elements

---

### 4. MVLookRenderer Node 🎨

**Purpose:** Apply machine vision aesthetics and color grading

**Key Features:**
- LUT (Look-Up Table) application
- Color grading presets (thermal, night vision, false color)
- Channel manipulation and mapping
- Edge enhancement and posterization
- GPU-accelerated color transforms

**Inputs:**
- Base image or layer
- LUT file or preset selection
- Intensity/mix control
- Additional processing options

**Outputs:**
- IMAGE with color grading applied

**Use Cases:**
- Thermal/infrared simulation
- Night vision aesthetics
- False color visualization
- Technical/scientific color mapping
- Artistic color grading
- Machine vision look development

---

## 🗓️ Implementation Timeline

### Priority Order (Recommended)

**Week 1-2: BoundingBoxRenderer**
- Most straightforward implementation
- Builds on existing Track Detect output
- High user value for detection workflows
- Tests pattern for other renderers

**Week 3-4: BlurRegionRenderer**
- Moderate complexity
- Requires region masking logic
- GPU optimization important for performance
- Useful for privacy and artistic effects

**Week 5-6: HUDTextRenderer**
- Text rendering complexity
- Font handling and layout
- Dynamic data formatting
- High value for technical users

**Week 7-8: MVLookRenderer**
- Most complex (LUT processing)
- Color space transformations
- Requires LUT file handling
- High creative value

### Alternative: Parallel Development

If multiple developers available:
- **Developer 1:** BBox + HUD (UI/overlay focus)
- **Developer 2:** Blur + MVLook (effects/color focus)

---

## 📋 Technical Requirements

### New Dependencies
```txt
# For text rendering (HUDTextRenderer)
pillow>=10.0.0  # Better text rendering than OpenCV

# For LUT processing (MVLookRenderer)
colour-science>=0.4.0  # Color space transformations
# OR implement LUT parsing manually
```

### GPU Considerations

All renderers should follow Phase 1 GPU patterns:
```python
class NewRenderer:
    def __init__(self):
        self.gpu = GPUAccelerator()  # Use existing GPU infrastructure

    def execute(self, **kwargs):
        if self.gpu.use_gpu:
            return self._execute_gpu(**kwargs)
        return self._execute_cpu(**kwargs)
```

---

## 🧪 Testing Strategy

### Per-Node Testing

**Each renderer needs:**
1. **Unit tests** - Core functionality
2. **Visual tests** - Output correctness
3. **Performance tests** - GPU benchmarks
4. **Integration tests** - With Phase 1 nodes

### Test Coverage Goals
- Unit test coverage: >80%
- Visual regression tests: All key features
- GPU performance: <10ms per operation @ 4K

---

## 📚 Documentation Updates Needed

### User Documentation
- Update README.md with 4 new nodes
- Add example workflows for each renderer
- Create combined workflow examples
- Update QUICK_START.md if needed

### Developer Documentation
- Update PROJECT_STATUS.md after each node
- Document new patterns/utilities added
- Add performance benchmarks to docs
- Update CHANGELOG.md

---

## 🎯 Success Criteria

### Phase 2 Complete When:

**Implementation:**
- ✅ All 4 nodes implemented and working
- ✅ GPU acceleration paths for all
- ✅ No performance regressions
- ✅ Clean integration with Phase 1 nodes

**Quality:**
- ✅ Unit tests passing (>80% coverage)
- ✅ Visual tests validating outputs
- ✅ Performance targets met (<10ms @ 4K)
- ✅ No memory leaks or GPU issues

**Documentation:**
- ✅ User guide updated
- ✅ Example workflows created
- ✅ API documented
- ✅ CHANGELOG updated

**Deployment:**
- ✅ Tested in ComfyUI
- ✅ No breaking changes to Phase 1
- ✅ All nodes visible in menu
- ✅ Workflows executing correctly

---

## 💡 Implementation Tips

### Reuse Phase 1 Infrastructure

**Don't reinvent:**
- Use existing `GPUAccelerator` class
- Reuse `image_utils` conversions
- Follow established node patterns
- Leverage `curve_math` if needed

**New utilities to create:**
- Text rendering helpers (for HUD)
- Region masking utilities (for Blur)
- LUT parsing/application (for MVLook)
- Box drawing primitives (for BBox)

### Code Organization

```
custom_nodes/ys_vision_tools/
├── nodes/
│   ├── bbox_renderer.py        # New
│   ├── blur_renderer.py        # New
│   ├── hud_renderer.py         # New
│   └── mvlook_renderer.py      # New
├── utils/
│   ├── text_utils.py           # New - Text rendering
│   ├── region_utils.py         # New - Region masking
│   ├── lut_utils.py            # New - LUT processing
│   └── [existing utils...]
└── tests/
    ├── unit/
    │   ├── test_bbox_renderer.py
    │   ├── test_blur_renderer.py
    │   ├── test_hud_renderer.py
    │   └── test_mvlook_renderer.py
    └── visual/
        └── [visual regression tests]
```

---

## 🚀 Getting Started

### Step 1: Review Phase 2 Plan
Read detailed implementation plan:
```
docs/plans/03-PHASE2-EXTENDED.md
```

### Step 2: Choose Starting Node
Recommended order: BBox → Blur → HUD → MVLook

### Step 3: Set Up Development Environment
Ensure Phase 1 environment is working:
```bash
# Test Phase 1 nodes first
cd custom_nodes/ys_vision_tools
python -c "from nodes import *; print('✓ Phase 1 imports OK')"
```

### Step 4: Follow TDD Approach
1. Write tests first (see Phase 2 plan for examples)
2. Implement node to pass tests
3. Add GPU acceleration
4. Benchmark performance
5. Document and integrate

---

## 📊 Progress Tracking

Track progress in PROJECT_STATUS.md:

```markdown
### Phase 2 Nodes (4 in progress)

1. **BoundingBoxRenderer** 📦
   - Status: [Not Started / In Progress / Complete]
   - Features: [list completed features]

2. **BlurRegionRenderer** 🌫️
   - Status: [Not Started / In Progress / Complete]
   - Features: [list completed features]

3. **HUDTextRenderer** 📊
   - Status: [Not Started / In Progress / Complete]
   - Features: [list completed features]

4. **MVLookRenderer** 🎨
   - Status: [Not Started / In Progress / Complete]
   - Features: [list completed features]
```

---

## 🎉 Phase 2 Vision

When Phase 2 is complete, YS-vision-tools will offer:

**10 Total Nodes:**
- ✅ 6 Phase 1 nodes (deployed)
- 🚧 4 Phase 2 nodes (upcoming)

**Complete Workflow Capabilities:**
```
Load Image
    ↓
Track Detect (Phase 1)
    ↓
    ├─→ Line Link Renderer (Phase 1)
    ├─→ Dot Renderer (Phase 1)
    ├─→ BBox Renderer (Phase 2) ← NEW!
    └─→ Blur Renderer (Phase 2) ← NEW!
    ↓
Palette Map (Phase 1)
    ↓
Layer Merge (Phase 1)
    ↓
    ├─→ HUD Renderer (Phase 2) ← NEW!
    └─→ MVLook Renderer (Phase 2) ← NEW!
    ↓
Composite Over (Phase 1)
    ↓
Preview/Save
```

---

**Ready to Begin:** ✅ Yes
**Phase 1 Status:** ✅ Complete & Deployed
**Detailed Plan:** See `docs/plans/03-PHASE2-EXTENDED.md`
**Questions?** Review CLAUDE.md for development guidelines
