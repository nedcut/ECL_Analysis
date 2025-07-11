"""ROI (Region of Interest) management for Brightness Sorcerer."""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Callable
import logging

from ..models.roi import ROI

# Constants
ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2
MOUSE_RESIZE_HANDLE_SENSITIVITY = 10


class ROIManager:
    """Manages ROI creation, editing, and rendering."""
    
    def __init__(self):
        self.rois: List[ROI] = []
        self.selected_roi_index: Optional[int] = None
        self.background_roi_index: Optional[int] = None
        
        # Interaction state
        self.drawing_mode = False
        self.moving_mode = False
        self.resizing_mode = False
        self.resize_corner: Optional[int] = None
        
        # Drawing state
        self.draw_start_point: Optional[Tuple[int, int]] = None
        self.draw_current_point: Optional[Tuple[int, int]] = None
        self.move_offset: Optional[Tuple[int, int]] = None
        
        # Callbacks
        self.on_roi_changed: Optional[Callable] = None
        self.on_selection_changed: Optional[Callable] = None
    
    def add_roi(self, x: int, y: int, width: int, height: int, 
                label: str = "", is_background: bool = False) -> int:
        """Add a new ROI and return its index."""
        try:
            # Get color for new ROI
            color_index = len(self.rois) % len(ROI_COLORS)
            color = ROI_COLORS[color_index]
            
            # Create ROI
            roi = ROI(
                x=x, y=y, width=width, height=height,
                color=color, label=label or f"ROI {len(self.rois) + 1}",
                is_background=is_background
            )
            
            self.rois.append(roi)
            roi_index = len(self.rois) - 1
            
            # Set as background ROI if specified
            if is_background:
                self.background_roi_index = roi_index
            
            # Select the new ROI
            self.selected_roi_index = roi_index
            
            # Notify of changes
            self._notify_roi_changed()
            self._notify_selection_changed()
            
            logging.info(f"Added ROI {roi_index}: {roi}")
            return roi_index
            
        except Exception as e:
            logging.error(f"Error adding ROI: {e}")
            return -1
    
    def add_roi_object(self, roi: ROI) -> int:
        """Add an existing ROI object and return its index."""
        try:
            self.rois.append(roi)
            roi_index = len(self.rois) - 1
            
            # Set as background ROI if specified
            if roi.is_background:
                self.background_roi_index = roi_index
            
            # Notify of changes
            self._notify_roi_changed()
            
            logging.info(f"Added ROI object {roi_index}: {roi}")
            return roi_index
            
        except Exception as e:
            logging.error(f"Error adding ROI object: {e}")
            return -1
    
    def delete_roi(self, index: int) -> bool:
        """Delete ROI at specified index."""
        if not self._is_valid_index(index):
            return False
        
        try:
            deleted_roi = self.rois.pop(index)
            
            # Update background ROI index
            if self.background_roi_index == index:
                self.background_roi_index = None
            elif self.background_roi_index is not None and self.background_roi_index > index:
                self.background_roi_index -= 1
            
            # Update selection
            if self.selected_roi_index == index:
                self.selected_roi_index = None
            elif self.selected_roi_index is not None and self.selected_roi_index > index:
                self.selected_roi_index -= 1
            
            self._notify_roi_changed()
            self._notify_selection_changed()
            
            logging.info(f"Deleted ROI {index}: {deleted_roi}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting ROI {index}: {e}")
            return False
    
    def select_roi(self, index: Optional[int]) -> bool:
        """Select ROI by index."""
        if index is not None and not self._is_valid_index(index):
            return False
        
        if self.selected_roi_index != index:
            self.selected_roi_index = index
            self._notify_selection_changed()
        
        return True
    
    def get_roi_at_point(self, x: int, y: int) -> Optional[int]:
        """Find ROI containing the specified point."""
        # Check in reverse order (top ROI first)
        for i in reversed(range(len(self.rois))):
            if self.rois[i].contains_point(x, y):
                return i
        return None
    
    def move_roi(self, index: int, new_x: int, new_y: int, 
                 frame_width: int, frame_height: int) -> bool:
        """Move ROI to new position within frame bounds."""
        if not self._is_valid_index(index):
            return False
        
        roi = self.rois[index]
        
        # Constrain to frame bounds
        new_x = max(0, min(new_x, frame_width - roi.width))
        new_y = max(0, min(new_y, frame_height - roi.height))
        
        roi.move_to(new_x, new_y)
        self._notify_roi_changed()
        return True
    
    def resize_roi(self, index: int, corner: int, new_x: int, new_y: int,
                   frame_width: int, frame_height: int) -> bool:
        """Resize ROI by dragging a corner."""
        if not self._is_valid_index(index):
            return False
        
        roi = self.rois[index]
        
        # Store original state for validation
        orig_x, orig_y = roi.x, roi.y
        orig_width, orig_height = roi.width, roi.height
        
        # Attempt resize
        roi.resize_from_corner(corner, new_x, new_y)
        
        # Validate bounds and minimum size
        if (roi.is_valid_for_frame(frame_width, frame_height) and 
            roi.width >= 10 and roi.height >= 10):
            self._notify_roi_changed()
            return True
        else:
            # Restore original state
            roi.x, roi.y = orig_x, orig_y
            roi.width, roi.height = orig_width, orig_height
            return False
    
    def set_background_roi(self, index: Optional[int]) -> bool:
        """Set or unset background ROI."""
        if index is not None and not self._is_valid_index(index):
            return False
        
        # Reset previous background ROI
        if self.background_roi_index is not None:
            self.rois[self.background_roi_index].is_background = False
        
        self.background_roi_index = index
        
        # Set new background ROI
        if index is not None:
            self.rois[index].is_background = True
        
        self._notify_roi_changed()
        return True
    
    def get_background_roi(self) -> Optional[ROI]:
        """Get the background ROI if set."""
        if self.background_roi_index is not None:
            return self.rois[self.background_roi_index]
        return None
    
    def render_rois(self, frame: np.ndarray) -> np.ndarray:
        """Render all ROIs on the frame."""
        if frame is None:
            return frame
        
        rendered_frame = frame.copy()
        
        for i, roi in enumerate(self.rois):
            thickness = (ROI_THICKNESS_SELECTED if i == self.selected_roi_index 
                        else ROI_THICKNESS_DEFAULT)
            
            # Draw ROI rectangle
            cv2.rectangle(rendered_frame, roi.top_left, roi.bottom_right, 
                         roi.color, thickness)
            
            # Draw label
            label_text = roi.label
            if roi.is_background:
                label_text += " (BG)"
            
            label_pos = (roi.x, roi.y - 5 if roi.y > 20 else roi.y + roi.height + 20)
            cv2.putText(rendered_frame, label_text, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE,
                       roi.color, ROI_LABEL_THICKNESS)
            
            # Draw corner handles for selected ROI
            if i == self.selected_roi_index:
                self._draw_corner_handles(rendered_frame, roi)
        
        # Draw current drawing ROI if in drawing mode
        if self.drawing_mode and self.draw_start_point and self.draw_current_point:
            self._draw_preview_roi(rendered_frame)
        
        return rendered_frame
    
    def _draw_corner_handles(self, frame: np.ndarray, roi: ROI):
        """Draw corner handles for resizing."""
        handles = roi.get_corner_handles()
        for handle_x, handle_y, handle_w, handle_h in handles:
            cv2.rectangle(frame, 
                         (handle_x, handle_y), 
                         (handle_x + handle_w, handle_y + handle_h),
                         (255, 255, 255), -1)
            cv2.rectangle(frame,
                         (handle_x, handle_y),
                         (handle_x + handle_w, handle_y + handle_h), 
                         roi.color, 2)
    
    def _draw_preview_roi(self, frame: np.ndarray):
        """Draw preview of ROI being drawn."""
        if not self.draw_start_point or not self.draw_current_point:
            return
        
        start_x, start_y = self.draw_start_point
        curr_x, curr_y = self.draw_current_point
        
        top_left = (min(start_x, curr_x), min(start_y, curr_y))
        bottom_right = (max(start_x, curr_x), max(start_y, curr_y))
        
        # Use next available color
        color_index = len(self.rois) % len(ROI_COLORS)
        color = ROI_COLORS[color_index]
        
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
    
    # Drawing interaction methods
    def start_drawing(self, x: int, y: int):
        """Start drawing a new ROI."""
        self.drawing_mode = True
        self.draw_start_point = (x, y)
        self.draw_current_point = (x, y)
    
    def update_drawing(self, x: int, y: int):
        """Update current drawing position."""
        if self.drawing_mode:
            self.draw_current_point = (x, y)
    
    def finish_drawing(self, x: int, y: int, frame_width: int, frame_height: int) -> int:
        """Finish drawing and create ROI."""
        if not self.drawing_mode or not self.draw_start_point:
            return -1
        
        start_x, start_y = self.draw_start_point
        
        # Calculate ROI bounds
        roi_x = min(start_x, x)
        roi_y = min(start_y, y)
        roi_width = abs(x - start_x)
        roi_height = abs(y - start_y)
        
        # Reset drawing state
        self.drawing_mode = False
        self.draw_start_point = None
        self.draw_current_point = None
        
        # Only create ROI if it has meaningful size
        if roi_width >= 10 and roi_height >= 10:
            return self.add_roi(roi_x, roi_y, roi_width, roi_height)
        
        return -1
    
    def cancel_drawing(self):
        """Cancel current drawing operation."""
        self.drawing_mode = False
        self.draw_start_point = None
        self.draw_current_point = None
    
    # Moving interaction methods
    def start_moving(self, roi_index: int, x: int, y: int) -> bool:
        """Start moving ROI."""
        if not self._is_valid_index(roi_index):
            return False
        
        roi = self.rois[roi_index]
        self.moving_mode = True
        self.move_offset = (x - roi.x, y - roi.y)
        return True
    
    def update_moving(self, x: int, y: int, frame_width: int, frame_height: int) -> bool:
        """Update ROI position while moving."""
        if not self.moving_mode or self.selected_roi_index is None or not self.move_offset:
            return False
        
        offset_x, offset_y = self.move_offset
        new_x = x - offset_x
        new_y = y - offset_y
        
        return self.move_roi(self.selected_roi_index, new_x, new_y, frame_width, frame_height)
    
    def finish_moving(self):
        """Finish moving ROI."""
        self.moving_mode = False
        self.move_offset = None
    
    # Resizing interaction methods
    def start_resizing(self, roi_index: int, corner: int) -> bool:
        """Start resizing ROI from corner."""
        if not self._is_valid_index(roi_index):
            return False
        
        self.resizing_mode = True
        self.resize_corner = corner
        return True
    
    def update_resizing(self, x: int, y: int, frame_width: int, frame_height: int) -> bool:
        """Update ROI size while resizing."""
        if (not self.resizing_mode or self.selected_roi_index is None or 
            self.resize_corner is None):
            return False
        
        return self.resize_roi(self.selected_roi_index, self.resize_corner, 
                              x, y, frame_width, frame_height)
    
    def finish_resizing(self):
        """Finish resizing ROI."""
        self.resizing_mode = False
        self.resize_corner = None
    
    def get_resize_corner_at_point(self, roi_index: int, x: int, y: int) -> Optional[int]:
        """Check if point is near a resize corner."""
        if not self._is_valid_index(roi_index):
            return None
        
        return self.rois[roi_index].get_resize_corner_from_point(x, y, MOUSE_RESIZE_HANDLE_SENSITIVITY)
    
    # State management
    def clear_all(self):
        """Clear all ROIs."""
        self.rois.clear()
        self.selected_roi_index = None
        self.background_roi_index = None
        self._reset_interaction_state()
        self._notify_roi_changed()
        self._notify_selection_changed()
    
    def _reset_interaction_state(self):
        """Reset all interaction state."""
        self.drawing_mode = False
        self.moving_mode = False
        self.resizing_mode = False
        self.resize_corner = None
        self.draw_start_point = None
        self.draw_current_point = None
        self.move_offset = None
    
    def cancel_all_interactions(self):
        """Cancel any ongoing interactions."""
        self._reset_interaction_state()
    
    # Validation and utilities
    def _is_valid_index(self, index: int) -> bool:
        """Check if ROI index is valid."""
        return 0 <= index < len(self.rois)
    
    def get_roi_count(self) -> int:
        """Get number of ROIs."""
        return len(self.rois)
    
    def get_roi(self, index: int) -> Optional[ROI]:
        """Get ROI by index."""
        if self._is_valid_index(index):
            return self.rois[index]
        return None
    
    def get_selected_roi(self) -> Optional[ROI]:
        """Get currently selected ROI."""
        if self.selected_roi_index is not None:
            return self.rois[self.selected_roi_index]
        return None
    
    # Serialization
    def to_dict(self) -> dict:
        """Convert ROI manager state to dictionary."""
        return {
            'rois': [roi.to_dict() for roi in self.rois],
            'selected_roi_index': self.selected_roi_index,
            'background_roi_index': self.background_roi_index
        }
    
    def from_dict(self, data: dict):
        """Load ROI manager state from dictionary."""
        self.clear_all()
        
        for roi_data in data.get('rois', []):
            roi = ROI.from_dict(roi_data)
            self.rois.append(roi)
        
        self.selected_roi_index = data.get('selected_roi_index')
        self.background_roi_index = data.get('background_roi_index')
        
        self._notify_roi_changed()
        self._notify_selection_changed()
    
    # Callbacks
    def _notify_roi_changed(self):
        """Notify that ROIs have changed."""
        if self.on_roi_changed:
            self.on_roi_changed()
    
    def _notify_selection_changed(self):
        """Notify that selection has changed."""
        if self.on_selection_changed:
            self.on_selection_changed()