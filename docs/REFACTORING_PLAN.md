# Brightness Sorcerer v2.0 - Module Decomposition Plan

## Current State
- Single file: `main.py` (~3,500 lines)
- Multiple classes mixed together
- No clear separation of concerns

## Proposed Module Structure

### Core Package Structure
```
brightness_sorcerer/
├── __init__.py
├── main.py (entry point only)
├── core/
│   ├── __init__.py
│   ├── video_processing.py
│   ├── analysis.py
│   ├── cache.py
│   └── exceptions.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── dialogs.py
│   ├── widgets.py
│   └── stylesheets.py
├── audio/
│   ├── __init__.py
│   ├── manager.py
│   └── analyzer.py
├── utils/
│   ├── __init__.py
│   ├── validation.py
│   ├── logging_config.py
│   └── constants.py
└── config/
    ├── __init__.py
    └── settings.py
```

### Module Responsibilities

#### core/video_processing.py
- `FrameCache` class
- Video file loading and validation
- Frame seeking and caching
- Video metadata extraction

#### core/analysis.py
- Brightness calculation algorithms
- ROI processing
- Statistical analysis
- Background subtraction
- `LowLightEnhancer` class

#### core/exceptions.py
- All custom exception classes
- Error handling utilities

#### ui/main_window.py
- `VideoAnalyzer` class (simplified)
- Main window layout and events
- ROI management UI

#### ui/dialogs.py
- `ProgressDialog` class
- Settings dialogs
- Error message dialogs

#### ui/widgets.py
- `StatusBar` class
- Custom UI components

#### audio/manager.py
- `AudioManager` class
- Audio playback controls

#### audio/analyzer.py
- `AudioAnalyzer` class
- Audio processing algorithms

#### utils/validation.py
- Input validation functions
- File validation utilities

#### utils/constants.py
- All application constants
- Color definitions
- Default values

#### utils/logging_config.py
- Logging setup and configuration

#### config/settings.py
- Settings management
- Configuration loading/saving

## Migration Strategy

### Phase 1: Extract Utility Modules (Completed Concepts)
- ✅ Create `utils/constants.py` with all constants
- ✅ Create `utils/validation.py` with validation functions
- ✅ Create `core/exceptions.py` with custom exceptions

### Phase 2: Extract Core Functionality
- Extract `FrameCache` to `core/cache.py`
- Extract brightness analysis to `core/analysis.py`
- Extract video processing to `core/video_processing.py`

### Phase 3: Extract UI Components
- Move dialogs to `ui/dialogs.py`
- Move widgets to `ui/widgets.py`
- Simplify main window in `ui/main_window.py`

### Phase 4: Extract Audio System
- Move `AudioManager` to `audio/manager.py`
- Move `AudioAnalyzer` to `audio/analyzer.py`

### Phase 5: Configuration Management
- Create centralized settings in `config/settings.py`
- Extract logging configuration to `utils/logging_config.py`

## Benefits of This Structure

### Maintainability
- Each module has single responsibility
- Clear boundaries between components
- Easier to locate and modify specific functionality

### Testability
- Each module can be unit tested independently
- Mock dependencies easily for testing
- Clear interfaces between components

### Reusability
- Core analysis algorithms can be reused
- UI components can be customized independently
- Audio system can be used in other projects

### Performance
- Modules can be lazy-loaded
- Memory usage optimization per module
- Better caching strategies

## Implementation Notes

### Import Strategy
- Use relative imports within packages
- Minimize circular dependencies
- Clear dependency hierarchy

### Backward Compatibility
- Maintain existing API during transition
- Gradual migration without breaking functionality
- Comprehensive testing at each phase

### Code Quality
- Add comprehensive type hints
- Include docstrings for all public APIs
- Follow PEP 8 coding standards
- Use consistent error handling patterns