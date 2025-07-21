# Testing Guide for Brightness Sorcerer

## Overview

This document provides comprehensive testing guidelines for the Brightness Sorcerer application, including test structure, execution procedures, and best practices.

## Test Architecture

### Test Types

1. **Unit Tests** (`tests/unit/`)
   - Test individual components in isolation
   - Fast execution (< 1 second each)
   - No external dependencies
   - High coverage of core logic

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Moderate execution time (1-10 seconds)
   - May use mock data/files
   - Focus on workflow validation

3. **GUI Tests** (marked with `@pytest.mark.gui`)
   - Test user interface components
   - Require display (may run headless)
   - Slower execution
   - User interaction workflows

### Test Structure

```
tests/
├── __init__.py                 # Test package
├── conftest.py                # Global fixtures and configuration
├── unit/                      # Unit tests
│   ├── test_exceptions.py     # Exception classes
│   ├── test_validation.py     # Validation functions
│   └── test_constants.py      # Constants and configuration
├── integration/               # Integration tests
│   └── test_main_integration.py  # Main application integration
└── fixtures/                  # Test data and utilities
    ├── create_test_data.py    # Test data generator
    └── [generated test files]  # Sample videos, images, etc.
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

Set up test data:
```bash
make setup-test-data
# or
python tests/fixtures/create_test_data.py
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest -m gui               # GUI tests only
pytest -m "not slow"        # Fast tests only
```

### Using Make Commands

```bash
make test           # Run all tests
make test-unit      # Unit tests only
make test-integration  # Integration tests only
make test-fast      # Skip slow tests
make test-coverage  # Run with coverage report
make test-smoke     # Quick smoke tests
```

## Test Configuration

### Pytest Configuration

The test suite is configured via `pytest.ini`:

- **Test Discovery**: Automatically finds `test_*.py` files
- **Markers**: Categorize tests (unit, integration, gui, slow, etc.)
- **Coverage**: Integrated coverage reporting
- **Output**: HTML and XML reports
- **Environment**: Headless GUI testing support

### Coverage Configuration

Coverage settings in `.coveragerc`:

- **Target Coverage**: 80% minimum
- **Source Files**: `brightness_sorcerer/` and `main.py`
- **Exclusions**: Test files, virtual environments
- **Reports**: HTML, XML, and terminal output

## Test Markers

Use markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.gui           # Requires GUI/display
@pytest.mark.slow          # Takes >5 seconds
@pytest.mark.video         # Requires video files
@pytest.mark.audio         # Requires audio functionality
@pytest.mark.performance   # Performance benchmark
@pytest.mark.smoke         # Basic smoke test
```

### Running Specific Markers

```bash
pytest -m unit              # Only unit tests
pytest -m "not slow"        # Exclude slow tests
pytest -m "gui and not slow" # GUI tests that are fast
```

## Writing Tests

### Test Naming Conventions

- **Files**: `test_[module_name].py`
- **Classes**: `Test[ComponentName]`
- **Methods**: `test_[specific_behavior]`

### Example Test Structure

```python
class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_video_file_success(self):
        """Test successful video file validation."""
        # Arrange
        valid_file = create_test_video()
        
        # Act
        result = validate_video_file(valid_file)
        
        # Assert
        assert result is True
    
    def test_validate_video_file_invalid_format(self):
        """Test validation fails for invalid format."""
        with pytest.raises(ValidationError, match="Unsupported format"):
            validate_video_file("test.txt")
```

### Using Fixtures

```python
def test_brightness_calculation(sample_image, sample_roi):
    """Test brightness calculation with fixtures."""
    # Use fixtures from conftest.py
    roi_image = extract_roi(sample_image, sample_roi)
    brightness = calculate_brightness(roi_image)
    assert 0 <= brightness <= 100
```

## Test Data and Fixtures

### Available Fixtures

Global fixtures defined in `conftest.py`:

- `qapp`: QApplication instance for GUI tests
- `sample_image`: Test image (RGB)
- `sample_lab_image`: Test image in LAB color space
- `sample_roi`: Sample ROI coordinates
- `mock_video_file`: Temporary test video file
- `sample_video_capture`: Mock video capture object
- `temp_config_dir`: Temporary configuration directory
- `sample_settings`: Sample application settings
- `mock_brightness_stats`: Mock brightness analysis results
- `test_data_generator`: Utility for generating test data

### Creating Test Data

```python
# Generate test video frames
generator = TestDataGenerator()
frames = generator.create_test_video_frames(num_frames=30)

# Generate brightness analysis data
brightness_data = generator.create_brightness_data(num_frames=100)
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.performance
def test_brightness_calculation_performance(benchmark, large_image):
    """Benchmark brightness calculation performance."""
    result = benchmark(calculate_brightness, large_image)
    assert result is not None
```

### Memory Testing

```python
def test_memory_usage_frame_cache():
    """Test frame cache memory usage."""
    cache = FrameCache(max_size=100)
    initial_memory = get_memory_usage()
    
    # Add frames to cache
    for i in range(100):
        cache.put(i, create_test_frame())
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Assert reasonable memory usage
    assert memory_increase < 100 * 1024 * 1024  # < 100MB
```

## GUI Testing

### Headless Testing

GUI tests run in headless mode by default:

```python
@pytest.mark.gui
def test_video_analyzer_initialization(qapp):
    """Test VideoAnalyzer class initialization."""
    from main import VideoAnalyzer
    
    analyzer = VideoAnalyzer()
    assert analyzer is not None
    assert analyzer.windowTitle() == "Brightness Sorcerer v2.0"
```

### User Interaction Testing

```python
@pytest.mark.gui
def test_roi_creation(qapp, qtbot):
    """Test ROI creation through user interaction."""
    from main import VideoAnalyzer
    
    analyzer = VideoAnalyzer()
    analyzer.show()
    
    # Simulate mouse clicks for ROI creation
    qtbot.mouseClick(analyzer.image_label, QtCore.Qt.LeftButton)
    qtbot.mouseDrag(analyzer.image_label, QtCore.Qt.LeftButton, 
                   QtCore.QPoint(10, 10), QtCore.QPoint(100, 100))
    
    # Verify ROI was created
    assert len(analyzer.rois) == 1
```

## Continuous Integration

### CI-Friendly Testing

```bash
# Run tests with CI-appropriate settings
make ci-test

# Or directly
pytest --junitxml=test-results.xml \
       --cov=brightness_sorcerer \
       --cov=main \
       --cov-report=xml \
       -m "not gui"  # Skip GUI tests in headless CI
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Unit Tests**: >90% coverage of core modules
- **Integration Tests**: >70% coverage of workflows
- **Critical Components**: 100% coverage required

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path includes project directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **GUI Test Failures**
   ```bash
   # Install virtual display for headless testing
   sudo apt-get install xvfb  # Ubuntu/Debian
   export QT_QPA_PLATFORM=offscreen
   ```

3. **Video Codec Issues**
   ```bash
   # Install required video codecs
   sudo apt-get install python3-opencv  # Ubuntu/Debian
   pip install opencv-python-headless   # Alternative
   ```

4. **Permission Errors**
   ```bash
   # Ensure test data directory is writable
   chmod 755 tests/fixtures/
   ```

### Debug Mode

Run tests with debugging output:

```bash
pytest -v -s --tb=long --log-cli-level=DEBUG
```

## Best Practices

### Test Design

1. **AAA Pattern**: Arrange, Act, Assert
2. **Isolation**: Each test should be independent
3. **Determinism**: Tests should produce consistent results
4. **Speed**: Unit tests should be fast (<1s each)
5. **Clarity**: Test names should describe expected behavior

### Mock Usage

```python
@patch('main.cv2.VideoCapture')
def test_video_loading(mock_video_capture, sample_video_capture):
    """Test video loading with mocked OpenCV."""
    mock_video_capture.return_value = sample_video_capture
    
    # Test video loading logic
    result = load_video("test.mp4")
    assert result is True
```

### Error Testing

```python
def test_error_handling():
    """Test that errors are properly handled."""
    with pytest.raises(ValidationError) as exc_info:
        validate_video_file("")
    
    assert "empty" in str(exc_info.value).lower()
```

## Quality Metrics

### Coverage Targets

- **Overall**: ≥80%
- **Core modules**: ≥90%
- **Validation functions**: 100%
- **Exception handling**: 100%

### Performance Targets

- **Unit tests**: <1 second each
- **Integration tests**: <10 seconds each
- **Full test suite**: <2 minutes
- **Memory usage**: <100MB during testing

## Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
make test-coverage
open htmlcov/index.html

# View terminal coverage summary
pytest --cov=brightness_sorcerer --cov-report=term-missing
```

### Test Reports

```bash
# Generate detailed HTML test report
make test-report
open test-report.html

# Generate JUnit XML for CI
pytest --junitxml=test-results.xml
```