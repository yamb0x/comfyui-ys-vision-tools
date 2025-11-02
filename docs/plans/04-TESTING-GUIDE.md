# Comprehensive Testing & Quality Assurance Guide

## 🎯 Testing Philosophy

**Core Principle**: Write tests FIRST (TDD), not after implementation.

**Why TDD?**
1. Forces clear specification before coding
2. Catches bugs immediately
3. Documents expected behavior
4. Makes refactoring safe
5. Builds confidence in code

## 📚 Testing Fundamentals for Engineers

### What is a Good Test?
A good test is:
- **Fast**: Runs in milliseconds, not seconds
- **Isolated**: Doesn't depend on other tests
- **Repeatable**: Same result every time
- **Self-Validating**: Pass/fail is obvious
- **Timely**: Written before the code

### Test Structure (AAA Pattern)
```python
def test_something():
    # Arrange - Set up test data
    input_data = create_test_data()

    # Act - Execute the function
    result = function_under_test(input_data)

    # Assert - Verify the result
    assert result == expected_value
```

## 🧪 Test Types & When to Use Them

### 1. Unit Tests
**Purpose**: Test individual functions/methods in isolation
**When**: For every new function you write
**Location**: `tests/unit/test_*.py`

```python
# Example: Testing a color mapping function
def test_map_id_to_color():
    # Test single ID
    color = map_id_to_color(42, palette=[[1,0,0], [0,1,0], [0,0,1]])
    assert len(color) == 3
    assert all(0 <= c <= 1 for c in color)

    # Test consistency (same ID = same color)
    color1 = map_id_to_color(42, palette)
    color2 = map_id_to_color(42, palette)
    assert np.array_equal(color1, color2)

    # Test edge cases
    assert map_id_to_color(0, palette) is not None
    assert map_id_to_color(-1, palette) is not None
    assert map_id_to_color(999999, palette) is not None
```

### 2. Integration Tests
**Purpose**: Test multiple components working together
**When**: After implementing related nodes
**Location**: `tests/integration/test_*.py`

```python
# Example: Testing tracking → rendering pipeline
def test_tracking_to_rendering_pipeline():
    # Create realistic test scenario
    test_frames = load_test_video_frames()

    # Run through pipeline
    tracks = track_detect_node.execute(test_frames[0])
    colors = palette_map_node.execute(tracks['ids'])
    layer = dot_renderer_node.execute(tracks['positions'], colors)

    # Verify output is valid
    assert layer.shape[2] == 4  # RGBA
    assert layer.max() <= 1.0
    assert layer.min() >= 0.0
```

### 3. Visual Tests
**Purpose**: Verify rendered output looks correct
**When**: For all rendering nodes
**Location**: `tests/visual/test_*.py`

```python
# Example: Visual regression test
def test_dot_renderer_visual():
    # Render dots
    layer = render_dots_at_positions([[50,50], [100,100]])

    # Save for manual inspection
    save_test_image(layer, "test_dots.png")

    # Automated checks
    assert has_content_at_position(layer, 50, 50)
    assert has_content_at_position(layer, 100, 100)
    assert is_mostly_transparent(layer)  # Most pixels should be transparent
```

### 4. Performance Tests
**Purpose**: Ensure code meets speed requirements
**When**: For computationally intensive functions
**Location**: `tests/performance/test_*.py`

```python
# Example: Performance benchmark
def test_track_detection_performance():
    image = np.random.rand(1080, 1920, 3)

    start = time.time()
    for _ in range(100):
        tracks = detect_tracks(image)
    elapsed = time.time() - start

    fps = 100 / elapsed
    assert fps > 10, f"Too slow: {fps:.1f} fps, need > 10 fps"
```

### 5. Property-Based Tests
**Purpose**: Test with random inputs to find edge cases
**When**: For functions with complex input spaces
**Tool**: `hypothesis` library

```python
from hypothesis import given, strategies as st

@given(
    width=st.integers(min_value=1, max_value=4096),
    height=st.integers(min_value=1, max_value=4096),
    num_points=st.integers(min_value=0, max_value=1000)
)
def test_renderer_handles_any_size(width, height, num_points):
    # Generate random points within bounds
    points = np.random.rand(num_points, 2) * [width, height]

    # Should not crash for any valid input
    layer = renderer.execute(points, width, height)

    # Basic sanity checks
    assert layer.shape == (height, width, 4)
    assert layer.dtype == np.float32
```

## 🔧 Testing Tools & Setup

### Required Testing Libraries
```bash
# Install testing dependencies
pip install pytest==7.4.3           # Test runner
pip install pytest-cov==4.1.0       # Coverage reporting
pip install pytest-mock==3.12.0     # Mocking support
pip install pytest-benchmark==4.0.0  # Performance testing
pip install hypothesis==6.92.1      # Property-based testing
pip install pytest-xdist==3.5.0     # Parallel test execution
```

### Pytest Configuration
Create `pytest.ini` in project root:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v                          # Verbose output
    --cov=custom_nodes/ys_vision # Coverage for our code
    --cov-report=term-missing   # Show missing lines
    --cov-report=html           # HTML coverage report
    --cov-fail-under=80         # Fail if coverage < 80%
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    visual: marks visual tests
    integration: marks integration tests
    benchmark: marks performance tests
```

### Test Fixtures
Create reusable test data in `tests/conftest.py`:
```python
import pytest
import numpy as np

@pytest.fixture
def sample_image():
    """Provide a standard test image"""
    return np.random.rand(480, 640, 3)

@pytest.fixture
def sample_tracks():
    """Provide standard track data"""
    return {
        'positions': np.array([[100, 100], [200, 200]]),
        'ids': np.array([1, 2]),
        'confidence': np.array([0.9, 0.8])
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary directory for test outputs"""
    return tmp_path / "test_output"
```

## 📝 Test Writing Patterns

### Pattern 1: Test Invalid Inputs
```python
def test_invalid_inputs():
    node = TrackDetectNode()

    # Test with None
    with pytest.raises(ValueError):
        node.execute(image=None)

    # Test with wrong shape
    with pytest.raises(ValueError):
        node.execute(image=np.array([1, 2, 3]))  # 1D array

    # Test with wrong type
    with pytest.raises(TypeError):
        node.execute(image="not_an_array")

    # Test with out-of-range values
    with pytest.raises(ValueError):
        node.execute(image=np.ones((100, 100, 3)) * 2.0)  # >1.0
```

### Pattern 2: Test Edge Cases
```python
def test_edge_cases():
    # Empty input
    result = process_tracks([])
    assert result == []

    # Single point
    result = process_tracks([[0, 0]])
    assert len(result) == 1

    # Duplicate points
    result = process_tracks([[50, 50], [50, 50]])
    assert handles_duplicates_correctly(result)

    # Points at image boundaries
    result = process_tracks([[0, 0], [639, 479]])
    assert all_points_valid(result)
```

### Pattern 3: Mock External Dependencies
```python
from unittest.mock import Mock, patch

def test_with_mocked_opencv():
    # Mock cv2 functions that might fail
    with patch('cv2.goodFeaturesToTrack') as mock_detect:
        mock_detect.return_value = np.array([[100, 100], [200, 200]])

        tracks = detect_corners(image)

        # Verify mock was called correctly
        mock_detect.assert_called_once()
        assert len(tracks) == 2
```

### Pattern 4: Parametrized Tests
```python
@pytest.mark.parametrize("blend_mode,expected", [
    ("normal", 0.5),
    ("add", 0.8),
    ("multiply", 0.3),
    ("screen", 0.65)
])
def test_blend_modes(blend_mode, expected):
    base = np.ones((10, 10, 4)) * 0.5
    overlay = np.ones((10, 10, 4)) * 0.6

    result = blend_layers(base, overlay, mode=blend_mode)
    assert abs(result[5, 5, 0] - expected) < 0.01
```

## 🎨 Visual Testing Strategy

### Creating Reference Images
```python
def create_reference_images():
    """Run once to create golden reference images"""

    # Create controlled test scenarios
    scenarios = [
        ("single_dot", [[100, 100]]),
        ("multiple_dots", [[50, 50], [150, 150], [250, 250]]),
        ("line_straight", [[0, 100], [200, 100]]),
        ("line_curved", [[0, 0], [100, 50], [200, 0]])
    ]

    for name, points in scenarios:
        layer = renderer.execute(points)
        save_image(layer, f"tests/fixtures/references/{name}.png")
```

### Comparing Against References
```python
def test_visual_regression():
    # Render current output
    current = renderer.execute(test_points)

    # Load reference
    reference = load_image("tests/fixtures/references/expected.png")

    # Calculate difference
    diff = np.abs(current - reference)
    max_diff = diff.max()

    # Allow small differences (e.g., floating point errors)
    assert max_diff < 0.01, f"Visual regression: max diff {max_diff}"

    # Save diff image for debugging
    if max_diff > 0:
        save_image(diff * 10, "test_diff.png")  # Amplify for visibility
```

## 🐛 Debugging Failed Tests

### 1. Use Pytest's Detailed Output
```bash
# Show local variables when test fails
pytest --showlocals

# Show full diff for assertions
pytest -vv

# Stop on first failure
pytest -x

# Enter debugger on failure
pytest --pdb
```

### 2. Add Debug Prints (temporarily)
```python
def test_something():
    result = complex_function()

    # Debug print (remove before commit)
    print(f"DEBUG: result shape = {result.shape}")
    print(f"DEBUG: result range = [{result.min()}, {result.max()}]")

    assert result.shape == (100, 100, 4)
```

### 3. Save Intermediate Results
```python
def test_complex_pipeline():
    # Save intermediate results for inspection
    step1 = process_step1(input_data)
    np.save("debug_step1.npy", step1)

    step2 = process_step2(step1)
    np.save("debug_step2.npy", step2)

    final = process_step3(step2)
    save_image(final, "debug_final.png")

    assert validate_output(final)
```

## 📊 Coverage Requirements

### Minimum Coverage by Component Type
- **Core algorithms**: 95% coverage required
- **Rendering functions**: 90% coverage required
- **Utility functions**: 85% coverage required
- **Node classes**: 80% coverage required
- **Integration points**: 75% coverage required

### Running Coverage Reports
```bash
# Run with coverage
pytest --cov=custom_nodes/ys_vision

# Generate HTML report
pytest --cov=custom_nodes/ys_vision --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac/Linux
start htmlcov/index.html  # Windows
```

### Excluding Code from Coverage
```python
# Mark code that shouldn't be covered
if TYPE_CHECKING:  # pragma: no cover
    from typing import SomeType

# Defensive code that should never execute
if this_should_never_happen:  # pragma: no cover
    raise RuntimeError("Impossible condition")
```

## 🏃 Continuous Testing Workflow

### Pre-Commit Testing
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Run fast tests before commit
pytest tests/unit -m "not slow" --quiet

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix before committing."
    exit 1
fi

echo "✅ Tests passed!"
```

### Test-Driven Development Cycle
```
1. Write failing test (Red)
   └─ Define expected behavior
   └─ Test fails (no implementation)

2. Write minimal code (Green)
   └─ Make test pass
   └─ Don't over-engineer

3. Refactor (Refactor)
   └─ Improve code quality
   └─ Tests still pass

4. Commit
   └─ git add -A
   └─ git commit -m "feat: implement X with tests"
```

## 🎯 Testing Checklist for Each Node

Before considering a node complete:

- [ ] Unit tests for all public methods
- [ ] Integration test with other nodes
- [ ] Visual test for rendered output
- [ ] Performance test meeting targets
- [ ] Edge case tests (empty, single, many)
- [ ] Invalid input tests
- [ ] Memory leak test (1000+ iterations)
- [ ] Coverage > 80%
- [ ] All tests pass
- [ ] No warnings from pytest

## 💡 Testing Tips for Success

1. **Test behavior, not implementation**
   - Bad: Test that function calls specific internal method
   - Good: Test that function produces correct output

2. **Keep tests simple**
   - Each test should test ONE thing
   - Test name should describe what's being tested

3. **Use meaningful assertions**
   ```python
   # Bad
   assert result

   # Good
   assert result is not None, "Function should return a value"
   assert len(result) == 3, f"Expected 3 items, got {len(result)}"
   ```

4. **Test the sad path too**
   - Don't just test success cases
   - Test failures, errors, edge cases

5. **Make tests independent**
   - Tests shouldn't depend on execution order
   - Each test should set up its own data

## Next Steps
Continue to `05-COMMON-PITFALLS.md` for troubleshooting guide.