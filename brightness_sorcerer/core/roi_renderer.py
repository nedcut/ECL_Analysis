"""ROI rendering system separated from business logic."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

from ..models.roi import ROI

# Rendering constants
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2
HANDLE_SIZE = 10


class ROIRenderer:
    """Handles visual rendering of ROIs on frames."""
    
    def __init__(self):
        self.show_labels = True
        self.show_handles = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render_rois(self, frame: np.ndarray, rois: List[ROI], 
                   selected_roi_index: Optional[int] = None,
                   preview_roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Render all ROIs on a frame.
        
        Args:
            frame: Input frame to render on
            rois: List of ROIs to render
            selected_roi_index: Index of selected ROI (for special rendering)
            preview_roi: Optional preview ROI (x, y, width, height) for drawing mode
            
        Returns:
            Frame with ROIs rendered
        """
        if frame is None:
            return frame
        
        rendered_frame = frame.copy()
        
        # Render regular ROIs
        for i, roi in enumerate(rois):
            is_selected = i == selected_roi_index
            self._render_single_roi(rendered_frame, roi, is_selected)
        
        # Render preview ROI if provided
        if preview_roi:
            self._render_preview_roi(rendered_frame, preview_roi)
        
        return rendered_frame
    
    def _render_single_roi(self, frame: np.ndarray, roi: ROI, is_selected: bool = False):
        """Render a single ROI on the frame."""
        thickness = ROI_THICKNESS_SELECTED if is_selected else ROI_THICKNESS_DEFAULT
        
        # Draw main rectangle
        cv2.rectangle(frame, roi.top_left, roi.bottom_right, roi.color, thickness)
        
        # Draw label if enabled
        if self.show_labels and roi.label:
            self._render_roi_label(frame, roi)
        
        # Draw resize handles if selected
        if is_selected and self.show_handles:
            self._render_resize_handles(frame, roi)
    
    def _render_roi_label(self, frame: np.ndarray, roi: ROI):
        """Render ROI label."""
        label_text = roi.label
        if roi.is_background:
            label_text += " (BG)"
        
        # Calculate label position (above ROI if possible, below if not enough space)
        label_pos = self._calculate_label_position(frame, roi, label_text)
        
        # Draw text background for better readability
        text_size = cv2.getTextSize(label_text, self.font, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS)[0]
        bg_top_left = (label_pos[0] - 2, label_pos[1] - text_size[1] - 4)
        bg_bottom_right = (label_pos[0] + text_size[0] + 2, label_pos[1] + 2)
        
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(frame, label_text, label_pos, self.font, 
                   ROI_LABEL_FONT_SCALE, roi.color, ROI_LABEL_THICKNESS)
    
    def _calculate_label_position(self, frame: np.ndarray, roi: ROI, text: str) -> Tuple[int, int]:
        """Calculate optimal position for ROI label."""
        text_size = cv2.getTextSize(text, self.font, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS)[0]
        
        # Try to place above ROI
        label_y = roi.y - 5
        
        # If not enough space above, place below
        if label_y - text_size[1] < 0:
            label_y = roi.y + roi.height + text_size[1] + 5
        
        # Ensure label doesn't go off screen horizontally
        label_x = max(0, min(roi.x, frame.shape[1] - text_size[0]))
        
        return (label_x, label_y)
    
    def _render_resize_handles(self, frame: np.ndarray, roi: ROI):
        """Render resize handles for selected ROI."""
        handles = roi.get_corner_handles(HANDLE_SIZE)
        
        for handle_x, handle_y, handle_w, handle_h in handles:
            # Draw white handle with colored border
            cv2.rectangle(frame, 
                         (handle_x, handle_y), 
                         (handle_x + handle_w, handle_y + handle_h),
                         (255, 255, 255), -1)  # White fill
            
            cv2.rectangle(frame,
                         (handle_x, handle_y),
                         (handle_x + handle_w, handle_y + handle_h), 
                         roi.color, 2)  # Colored border
    
    def _render_preview_roi(self, frame: np.ndarray, preview_roi: Tuple[int, int, int, int]):
        """Render preview ROI during drawing."""
        x, y, width, height = preview_roi
        
        if width <= 0 or height <= 0:
            return
        
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        
        # Use a semi-transparent style for preview
        color = (255, 255, 0)  # Yellow for preview
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        
        # Draw dashed lines for preview effect
        self._draw_dashed_rectangle(frame, top_left, bottom_right, color)
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, 
                              top_left: Tuple[int, int], 
                              bottom_right: Tuple[int, int], 
                              color: Tuple[int, int, int]):
        """Draw a dashed rectangle."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        dash_length = 5
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 1)
        
        # Bottom edge  
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 1)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 1)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 1)
    
    def set_rendering_options(self, show_labels: bool = True, show_handles: bool = True):
        """Configure rendering options."""
        self.show_labels = show_labels
        self.show_handles = show_handles
        logging.debug(f"ROI rendering options updated: labels={show_labels}, handles={show_handles}")
    
    def create_roi_overlay(self, frame_shape: Tuple[int, int], rois: List[ROI], 
                          selected_roi_index: Optional[int] = None) -> np.ndarray:
        """
        Create an overlay image with just the ROIs (no background frame).
        
        Args:
            frame_shape: Shape of the target frame (height, width)
            rois: List of ROIs to render
            selected_roi_index: Index of selected ROI
            
        Returns:
            Overlay image with transparent background
        """
        # Create transparent overlay
        height, width = frame_shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
        
        for i, roi in enumerate(rois):
            is_selected = i == selected_roi_index
            thickness = ROI_THICKNESS_SELECTED if is_selected else ROI_THICKNESS_DEFAULT
            
            # Convert BGR color to RGBA
            color_rgba = (*roi.color, 255)
            
            # Draw rectangle on overlay
            cv2.rectangle(overlay, roi.top_left, roi.bottom_right, color_rgba, thickness)
            
            # Draw handles if selected
            if is_selected and self.show_handles:
                self._render_handles_on_overlay(overlay, roi, color_rgba)
        
        return overlay
    
    def _render_handles_on_overlay(self, overlay: np.ndarray, roi: ROI, color: Tuple[int, int, int, int]):
        """Render resize handles on overlay."""
        handles = roi.get_corner_handles(HANDLE_SIZE)
        
        for handle_x, handle_y, handle_w, handle_h in handles:
            # White handle with colored border
            cv2.rectangle(overlay, 
                         (handle_x, handle_y), 
                         (handle_x + handle_w, handle_y + handle_h),
                         (255, 255, 255, 255), -1)
            
            cv2.rectangle(overlay,
                         (handle_x, handle_y),
                         (handle_x + handle_w, handle_y + handle_h), 
                         color, 2)
    
    def get_roi_bounds_image(self, frame_shape: Tuple[int, int], roi: ROI) -> np.ndarray:
        """
        Create a mask image showing the bounds of a single ROI.
        
        Args:
            frame_shape: Shape of the target frame
            roi: ROI to create mask for
            
        Returns:
            Binary mask image
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill ROI area with white
        cv2.rectangle(mask, roi.top_left, roi.bottom_right, 255, -1)
        
        return mask