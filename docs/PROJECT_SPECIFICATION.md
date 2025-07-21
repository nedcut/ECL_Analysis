# Brightness Sorcerer v2.0 - Project Specification

## Executive Summary

Brightness Sorcerer v2.0 is a professional desktop application for advanced video brightness analysis. The software enables researchers, engineers, and analysts to precisely measure brightness changes in user-defined regions of interest (ROIs) within video content using scientifically accurate CIE LAB color space measurements.

## Project Goals

### Primary Objectives

1. **Scientific Accuracy**: Provide perceptually uniform brightness measurements using CIE LAB L* channel (0-100 scale) for consistent, human-vision-aligned analysis
2. **Interactive Analysis**: Enable intuitive ROI definition and manipulation directly on video frames with real-time feedback
3. **Automated Detection**: Implement intelligent frame range detection based on brightness thresholds and statistical analysis
4. **Professional Output**: Generate publication-quality visualizations and comprehensive data exports for research and reporting
5. **User Experience**: Deliver smooth, responsive video navigation with efficient frame caching and keyboard shortcuts

### Secondary Objectives

1. **Blue Light Analysis**: Provide dedicated blue channel analysis (0-255 scale) for specialized applications
2. **Background Correction**: Support background ROI subtraction for noise reduction and baseline normalization
3. **Performance Optimization**: Maintain responsive performance with large video files through intelligent caching
4. **Data Integrity**: Ensure robust error handling and validation throughout the analysis pipeline
5. **Extensibility**: Design modular architecture for future enhancements and feature additions

## Core Functionality

### Video Processing & Analysis

#### Supported Input Formats
- **Video Formats**: MP4, MOV, AVI, MKV, WMV, M4V, FLV
- **Resolution Range**: 32x32 to 8K (7680x4320)
- **Duration Range**: 0.1 seconds to 2 hours
- **Frame Rate**: Adaptive support with 30fps fallback

#### Brightness Measurement
- **Color Space**: CIE LAB conversion for perceptually uniform measurements
- **L* Channel**: Primary brightness analysis (0-100 scale)
- **Blue Channel**: Dedicated blue light analysis (0-255 scale)
- **Statistics**: Dual calculation of mean and median values for robust analysis
- **Noise Filtering**: Configurable threshold filtering (default: 5 L* units)

#### Low-Light Enhancement
- **Bilateral Filtering**: Edge-preserving noise reduction
- **Adaptive Histogram Equalization**: CLAHE for improved contrast
- **Channel Boosting**: Configurable L* and blue channel amplification
- **Signal Processing**: Dynamic range compression and signal amplification

### Region of Interest (ROI) Management

#### Interactive ROI System
- **Capacity**: Up to 8 color-coded ROIs per analysis
- **Operations**: Draw, select, move, resize, delete with mouse interaction
- **Visual Feedback**: Color-coded rectangles with selection handles
- **Background ROI**: Special designation for baseline subtraction
- **Validation**: Automatic boundary checking and size constraints

#### ROI Features
- **Real-time Display**: Live brightness values for current frame
- **Statistical Output**: Mean and median calculations for both L* and blue channels
- **Background Subtraction**: Automatic baseline correction using background ROI
- **Persistence**: ROI configurations saved with analysis settings

### Automated Frame Detection

#### Auto-Detection Algorithm
- **Baseline Calculation**: 5th percentile brightness across entire video
- **Threshold Setting**: Configurable offset above baseline (default: 5.0 L*)
- **Frame Range**: Automatic start/end frame detection based on brightness peaks
- **Manual Override**: User-defined thresholds and frame ranges supported

#### Analysis Range Control
- **Automatic Mode**: Intelligent detection of relevant frame ranges
- **Manual Mode**: User-specified start and end frames
- **Validation**: Range checking and boundary enforcement
- **Preview**: Real-time threshold visualization

### Data Export & Visualization

#### CSV Data Export
- **Frame-by-Frame Data**: Complete temporal brightness series
- **Dual Statistics**: Mean and median values for comprehensive analysis
- **Multi-Channel**: Both L* and blue channel measurements
- **Background Correction**: Raw and background-subtracted values
- **Timestamp Information**: Frame timing and metadata

#### Enhanced Plotting System
- **Dual-Panel Plots**: Separate L* and blue channel visualization
- **Statistical Overlays**: Confidence bands and variability indicators
- **Peak Detection**: Automatic identification and marking of brightness peaks
- **Professional Quality**: 300 DPI output with publication-ready formatting
- **Customizable Styling**: Configurable colors, fonts, and layout options

### User Interface Design

#### Modern GUI Framework
- **Technology**: PyQt5-based desktop application
- **Theme**: Dark mode interface with high contrast elements
- **Responsiveness**: Smooth resizing and layout adaptation
- **Accessibility**: Keyboard navigation and screen reader compatibility

#### Navigation & Controls
- **Video Playback**: Frame-by-frame navigation with caching
- **Keyboard Shortcuts**: Comprehensive hotkey system for efficient workflow
- **Drag & Drop**: Simple file loading interface
- **Recent Files**: Quick access to previously analyzed videos

#### Real-time Feedback
- **Brightness Display**: Live ROI brightness values for current frame
- **Progress Tracking**: Analysis progress with ETA estimation
- **Status Updates**: Real-time feedback during long operations
- **Error Handling**: Clear error messages and recovery guidance

## Technical Architecture

### Core Components

#### Application Structure
- **Single-File Design**: Complete application in main.py (~2200+ lines)
- **Modular Classes**: Separate concerns with FrameCache and VideoAnalyzer classes
- **Configuration Management**: JSON-based settings persistence
- **Error Handling**: Comprehensive exception handling and logging

#### Performance Optimization
- **Frame Caching**: LRU cache system (100 frame default) for smooth navigation
- **Memory Management**: Automatic cleanup and resource management
- **Efficient Seeking**: Smart frame reading with minimal file access
- **Progress Monitoring**: Real-time performance tracking and optimization

#### Data Processing Pipeline
```
Video Input → Frame Extraction → Color Space Conversion → 
ROI Processing → Statistical Analysis → Background Correction → 
Data Export → Visualization Generation
```

### Quality Assurance

#### Validation System
- **Input Validation**: Comprehensive video file and parameter checking
- **Range Validation**: ROI boundary and size constraint enforcement
- **Data Integrity**: Statistical calculation verification and error detection
- **Output Validation**: Export format verification and quality checking

#### Error Recovery
- **Graceful Degradation**: Fallback mechanisms for failed operations
- **Resource Cleanup**: Automatic memory and file handle management
- **User Guidance**: Clear error messages with recovery suggestions
- **Logging System**: Comprehensive debugging and audit trail

## Development Standards

### Code Quality
- **Type Hints**: Complete type annotation throughout codebase
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Robust exception handling with specific error types
- **Testing**: Manual testing protocols for all major functionality

### Performance Requirements
- **Responsiveness**: Sub-100ms UI response times for user interactions
- **Memory Efficiency**: Optimal memory usage with configurable cache sizes
- **Scalability**: Support for large video files (up to 2 hours duration)
- **Accuracy**: Scientifically accurate brightness measurements with validation

### Configuration Management
- **Settings Persistence**: Automatic save/restore of user preferences
- **Backup Systems**: Configuration backup and recovery mechanisms
- **Default Values**: Sensible defaults for all configurable parameters
- **Validation**: Settings validation and migration support

## Future Enhancement Opportunities

### Planned Features
1. **Batch Processing**: Multiple video analysis with queue management
2. **Plugin Architecture**: Extensible analysis module system
3. **Cloud Integration**: Remote storage and collaboration features
4. **Advanced Statistics**: Additional statistical measures and algorithms
5. **API Interface**: Programmatic access for automation and integration

### Technical Improvements
1. **Multi-threading**: Parallel processing for improved performance
2. **GPU Acceleration**: Hardware acceleration for intensive operations
3. **Database Integration**: Structured data storage and querying
4. **Network Protocols**: Remote video analysis and streaming support
5. **Machine Learning**: Automated pattern recognition and classification

## Success Criteria

### Functional Requirements
- ✅ Load and display video files in supported formats
- ✅ Interactive ROI definition and manipulation
- ✅ Accurate CIE LAB brightness measurements
- ✅ Automated frame range detection
- ✅ CSV data export with comprehensive statistics
- ✅ Professional-quality plot generation
- ✅ Real-time brightness feedback
- ✅ Settings persistence and recent files management

### Performance Requirements
- ✅ Smooth video navigation with frame caching
- ✅ Sub-second analysis startup for typical videos
- ✅ Memory usage under 1GB for standard operations
- ✅ Support for videos up to 2 hours duration
- ✅ Responsive UI during analysis operations

### Quality Requirements
- ✅ Scientifically accurate brightness measurements
- ✅ Robust error handling and recovery
- ✅ Professional user interface design
- ✅ Comprehensive data validation
- ✅ Publication-quality output generation

## Conclusion

Brightness Sorcerer v2.0 represents a comprehensive solution for professional video brightness analysis, combining scientific accuracy with user-friendly design. The application successfully delivers on its core objectives of providing perceptually uniform brightness measurements, interactive analysis capabilities, and professional-quality output suitable for research and commercial applications.

The modular architecture and comprehensive feature set position the software for continued development and enhancement, while the current implementation provides a robust foundation for immediate productive use in brightness analysis workflows.