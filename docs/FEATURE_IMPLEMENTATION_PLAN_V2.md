# Feature Implementation Plan v2

This document outlines the plan to implement two new features: best-fit frame analysis and fixing the threshold-based pixel mask.

## Feature 1: Best-Fit Frame Analysis

**Objective:** Implement a mechanism to find the "best" frame (e.g., the one with the highest brightness in a user-selected ROI) and use the ROI from that frame as a template for analyzing all other frames in the selected range.

**Implementation Steps:**

1.  **UI Changes:**
    *   Add a new button to the UI, labeled "Find & Lock Best ROI". This button will be placed in the "Regions of Interest (ROI)" group box.
    *   Add a visual indicator (e.g., a label or icon) to show when a "locked" ROI is active.

2.  **Backend Logic:**
    *   Create a new function, `find_and_lock_best_roi`.
    *   This function will first check if there is a selected ROI.
    *   It will then iterate through the frames specified by the "Run Duration" settings.
    *   For each frame, it will calculate the mean brightness of the selected ROI.
    *   The frame with the highest mean brightness will be identified as the "best" frame.
    *   The position and size of the ROI from this "best" frame will be stored as the "locked" ROI.

3.  **Analysis Integration:**
    *   Modify the main analysis loop (`analyze_video` function).
    *   If a "locked" ROI is active, the analysis will use the locked ROI's coordinates for every frame in the analysis range.
    *   If no "locked" ROI is active, the analysis will proceed as it currently does (using the ROI's position on each frame).

## Feature 2: Fix Threshold-Based Masking

**Objective:** Correct the pixel mask visualization so that it accurately reflects the currently active threshold (either the manual L* value or the calculated background ROI threshold).

**Implementation Steps:**

1.  **Code Modification:**
    *   Locate the `_apply_pixel_mask_overlay` function in `main.py`.
    *   Inside this function, modify the logic to correctly retrieve the active threshold value.
    *   If a background ROI is set, the `background_brightness` should be calculated from that ROI for the current frame.
    *   If no background ROI is set, the `background_brightness` should be set to the value of the `manual_threshold` spinbox.

2.  **Dynamic Updates:**
    *   Ensure that any change to the manual threshold spinbox or the selection of a new background ROI immediately triggers a redraw of the pixel mask on the currently displayed frame.
    *   This will be achieved by connecting the `valueChanged` signal of the threshold spinbox and the `clicked` signal of the "Set Selected ROI as Background" button to a function that calls `show_frame`.

## To-Do List

- [x] **Feature 1: Best-Fit Frame Analysis**
    - [x] Add "Find & Lock Best ROI" button to the UI.
    - [x] Implement `find_and_lock_best_roi` function.
    - [x] Integrate "locked" ROI logic into the `analyze_video` function.
    - [x] Add UI indicator for "locked" ROI state.

- [x] **Feature 2: Fix Threshold-Based Masking**
    - [x] Modify `_apply_pixel_mask_overlay` to use the correct threshold.
    - [x] Implement dynamic updates for the pixel mask when the threshold changes.

- [ ] **Testing**
    - [ ] Test Feature 1 with various videos and ROIs.
    - [ ] Test Feature 2 to confirm the mask updates correctly with both manual and background ROI thresholds.