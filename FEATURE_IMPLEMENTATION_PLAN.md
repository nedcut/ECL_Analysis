# Feature Implementation Plan - Brightness Sorcerer v3.0

## Overview
This document outlines the implementation plan for new features to enhance the Brightness Sorcerer application with improved analysis capabilities, better visualization, and sound integration.

## Feature 1: Background L* Subtraction (Unthresholded)

### Current State
- Background ROI system exists but only used for threshold calculation
- Current analysis applies noise floor filtering (removes pixels < 5 L*)

### Implementation Plan

#### Changes to `_compute_brightness_stats()` (main.py:2139)
1. **Add new parameter**: `background_brightness: Optional[float] = None`
2. **Remove threshold filtering**: Calculate stats on all pixels in ROI
3. **Apply background subtraction**: If background_brightness provided, subtract from L* values before computing mean/median
4. **Clamp negative values**: Ensure subtracted values don't go below 0

#### Changes to Analysis Pipeline (`analyze_video()` - main.py:1825)
1. **Calculate background per frame**: If background ROI defined, compute its mean brightness for each frame
2. **Pass background to ROI analysis**: Supply background brightness to each ROI's stats calculation
3. **Update CSV output**: Add columns for raw brightness and background-subtracted brightness

#### New Methods
```python
def _compute_background_brightness(self, frame: np.ndarray) -> Optional[float]:
    """Calculate background ROI brightness for current frame"""
    
def _compute_brightness_stats_with_background(self, roi_bgr: np.ndarray, background_brightness: Optional[float]) -> Tuple[float, float, float, float]:
    """Return (raw_mean, raw_median, bg_sub_mean, bg_sub_median)"""
```

**Estimated Effort**: 4-6 hours

---

## Feature 2: Pixel Visualization During Analysis

### Current State
- ROIs shown as colored rectangles during navigation
- No visualization of which pixels are being analyzed

### Implementation Plan

#### New Visualization Component
1. **Add checkbox**: "Show Pixel Mask" in analysis controls
2. **Create overlay rendering**: Method to highlight analyzed pixels in red
3. **Update during analysis**: Refresh visualization every N frames during analysis

#### Changes to `_draw_rois()` (main.py:1083)
1. **Add mask overlay mode**: When enabled, show red overlay on pixels above noise floor
2. **Use transparency**: Alpha blending to show original video underneath
3. **Real-time toggle**: Allow enabling/disabling during analysis

#### New Methods
```python
def _create_pixel_mask_overlay(self, frame: np.ndarray, roi_coords: Tuple) -> np.ndarray:
    """Create red overlay showing which pixels are analyzed"""
    
def _apply_mask_visualization(self, frame: np.ndarray) -> np.ndarray:
    """Apply pixel mask overlay to frame if enabled"""
```

**Estimated Effort**: 3-4 hours

---

## Feature 3: Remove Mean-Median Difference Graph

### Current State
- `_generate_enhanced_plot()` creates dual-panel plots with difference graph

### Implementation Plan

#### Simplify Plot Generation
1. **Remove second subplot**: Eliminate ax2 (difference plot) from `_generate_enhanced_plot()`
2. **Expand main plot**: Use full figure height for brightness trends
3. **Keep statistics**: Maintain statistical overlays and annotations
4. **Update plot layout**: Adjust figure size and spacing

#### Modified Plot Structure
- Single panel with mean and median trends
- Statistical annotations (peaks, averages, confidence bands)
- Cleaner, more focused visualization

**Estimated Effort**: 1-2 hours

---

## Feature 4: Enhanced Side Panel Information

### Current State
- Basic brightness display in right panel
- Threshold settings in separate groupbox

### Implementation Plan

#### New Information Panel
1. **Consolidate displays**: Create comprehensive "Current Frame Info" panel
2. **Real-time updates**: Update on every frame change
3. **Structured layout**: Organized sections for different data types

#### Information to Display
```
Current Frame Info
├── Frame: 1234 / 5678 (21.8%)
├── Brightness (Raw)
│   ├── ROI 1: Mean 45.2, Median 43.8
│   ├── ROI 2: Mean 52.1, Median 51.9
│   └── Background: 15.3
├── Brightness (Background Subtracted)
│   ├── ROI 1: Mean 29.9, Median 28.5
│   └── ROI 2: Mean 36.8, Median 36.6
├── Analysis Stats
│   ├── Current Threshold: 20.3 L*
│   ├── Above Threshold: ROI 1, ROI 2
│   └── Analysis Range: 1200-4500
└── Preview Stats (±std)
    ├── ROI 1: 29.9 ± 12.4
    └── ROI 2: 36.8 ± 8.7
```

#### New Methods
```python
def _create_info_panel(self) -> QtWidgets.QGroupBox:
    """Create comprehensive current frame information panel"""
    
def _update_frame_info_display(self):
    """Update all information in the side panel"""
    
def _calculate_preview_stats(self) -> Dict[int, Tuple[float, float]]:
    """Calculate running statistics for preview display"""
```

**Estimated Effort**: 5-6 hours

---

## Feature 5: Sound Implementation with Run Duration

### Current State
- No audio feedback system
- No run duration tracking

### Implementation Plan

#### Audio System
1. **Add dependencies**: `pygame` or `pydub` for cross-platform audio
2. **Sound preferences**: User-configurable beep type and volume
3. **Audio triggers**: Beep at analysis start, end, and run detection

#### Run Duration Integration
1. **New input field**: "Expected Run Duration (seconds)" in analysis controls
2. **Smart detection**: Use duration to validate auto-detected ranges
3. **Audio markers**: Beep when detecting run start/end within expected duration

#### Changes to Auto-Detection Algorithm
1. **Duration validation**: Prefer detected ranges close to expected duration
2. **Multiple candidate filtering**: If multiple bright periods found, choose best match
3. **Confidence scoring**: Rate detection quality based on duration match

#### New Components
```python
class AudioManager:
    """Handle all audio feedback in the application"""
    def play_beep(self, frequency: int = 800, duration: int = 200): ...
    def play_analysis_start(self): ...
    def play_analysis_complete(self): ...
    def play_run_detected(self): ...

def _validate_run_duration(self, start_frame: int, end_frame: int, expected_duration: float) -> float:
    """Calculate confidence score for detected run vs expected duration"""
```

#### UI Additions
- Run duration input field (QDoubleSpinBox)
- Audio settings (enable/disable, volume)
- Duration validation feedback

**Estimated Effort**: 6-8 hours

---

## Feature 6: PSTrace Integration Enhancement

### Current State
- Basic auto-detection based on brightness thresholds
- No external synchronization

### Implementation Plan

#### Enhanced Auto-Detection
1. **Temporal analysis**: Look for consistent bright periods matching expected duration
2. **Multiple run detection**: Identify and rank multiple candidate runs
3. **Timing precision**: Frame-accurate start/end detection with sub-frame interpolation

#### Integration Features
1. **Export timing data**: CSV with precise timestamps for synchronization
2. **Validation tools**: Visual indicators of detection confidence
3. **Manual override**: Easy adjustment of auto-detected boundaries

#### New Methods
```python
def _detect_multiple_runs(self, expected_duration: float) -> List[Tuple[int, int, float]]:
    """Return list of (start, end, confidence) for potential runs"""
    
def _export_timing_data(self, runs: List[Tuple[int, int, float]], save_dir: str):
    """Export timing information for external synchronization"""
```

**Estimated Effort**: 4-5 hours

---

## Implementation Priority and Timeline

### Phase 1: Core Analysis Improvements (Week 1)
1. **Background L* subtraction** (Priority: High)
2. **Remove mean-median difference graph** (Priority: Medium)
3. **Pixel visualization during analysis** (Priority: Medium)

### Phase 2: UI Enhancements (Week 2)
1. **Enhanced side panel information** (Priority: High)
2. **Sound implementation with run duration** (Priority: Medium)

### Phase 3: Integration Features (Week 3)
1. **PSTrace integration enhancement** (Priority: Low)
2. **Testing and refinement** (Priority: High)

## Technical Requirements

### New Dependencies
```
# Add to requirements.txt
pygame>=2.0.0  # For cross-platform audio
```

### Architecture Changes
- New `AudioManager` class for sound handling
- Enhanced information display system
- Modified analysis pipeline for background subtraction
- Improved auto-detection algorithms

### Backward Compatibility
- Maintain existing CSV format with additional columns
- Preserve current analysis workflow
- Keep existing keyboard shortcuts and UI behavior

## Testing Strategy

### Unit Tests
- Background subtraction calculations
- Audio system functionality
- Run duration validation algorithms

### Integration Tests
- Full analysis pipeline with new features
- UI responsiveness during long analyses
- Audio timing and synchronization

### User Acceptance Tests
- Workflow efficiency improvements
- Visual feedback clarity
- Sound system usability

## Risk Assessment

### Technical Risks
- **Audio system compatibility**: Different behavior across OS platforms
- **Performance impact**: Pixel visualization may slow analysis
- **Memory usage**: Enhanced information display and pixel masks

### Mitigation Strategies
- Thorough testing on multiple platforms
- Optional visualization features with performance monitoring
- Efficient memory management and cleanup

---

**Total Estimated Effort**: 20-25 hours
**Recommended Development Approach**: Iterative implementation with user feedback after each phase