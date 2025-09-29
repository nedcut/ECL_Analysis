"""ROI (Region of Interest) management module."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..constants import (
    ROI_COLORS,
    ROI_THICKNESS_DEFAULT,
    ROI_THICKNESS_SELECTED,
    ROI_LABEL_FONT_SCALE,
    ROI_LABEL_THICKNESS
)


class ROIManager:
    """Manages regions of interest for video analysis."""

    def __init__(self):
        """Initialize ROI manager."""
        self.rois: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.selected_roi_idx: Optional[int] = None
        self.background_roi_idx: Optional[int] = None

        # Drawing state
        self.is_drawing = False
        self.is_moving = False
        self.is_resizing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.move_offset: Optional[Tuple[int, int]] = None

    def add_roi(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> int:
        """
        Add a new ROI.

        Args:
            pt1: First corner (x, y)
            pt2: Second corner (x, y)

        Returns:
            Index of the newly added ROI
        """
        # Normalize points so pt1 is top-left and pt2 is bottom-right
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])

        self.rois.append(((x1, y1), (x2, y2)))
        return len(self.rois) - 1

    def remove_roi(self, index: int) -> bool:
        """
        Remove an ROI by index.

        Args:
            index: Index of ROI to remove

        Returns:
            True if successfully removed, False otherwise
        """
        if 0 <= index < len(self.rois):
            self.rois.pop(index)

            # Adjust selected index
            if self.selected_roi_idx == index:
                self.selected_roi_idx = None
            elif self.selected_roi_idx is not None and self.selected_roi_idx > index:
                self.selected_roi_idx -= 1

            # Adjust background ROI index
            if self.background_roi_idx == index:
                self.background_roi_idx = None
            elif self.background_roi_idx is not None and self.background_roi_idx > index:
                self.background_roi_idx -= 1

            return True
        return False

    def clear_rois(self) -> None:
        """Remove all ROIs."""
        self.rois.clear()
        self.selected_roi_idx = None
        self.background_roi_idx = None

    def get_roi_count(self) -> int:
        """Get the number of defined ROIs."""
        return len(self.rois)

    def get_roi(self, index: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get ROI coordinates by index.

        Args:
            index: ROI index

        Returns:
            ((x1, y1), (x2, y2)) or None if index is invalid
        """
        if 0 <= index < len(self.rois):
            return self.rois[index]
        return None

    def set_selected_roi(self, index: Optional[int]) -> None:
        """
        Set the selected ROI index.

        Args:
            index: ROI index to select, or None to deselect
        """
        if index is None or 0 <= index < len(self.rois):
            self.selected_roi_idx = index

    def set_background_roi(self, index: Optional[int]) -> bool:
        """
        Set the background ROI index.

        Args:
            index: ROI index to use as background, or None to clear

        Returns:
            True if successfully set, False if index is invalid
        """
        if index is None:
            self.background_roi_idx = None
            return True

        if 0 <= index < len(self.rois):
            self.background_roi_idx = index
            return True

        return False

    def draw_rois(
        self,
        frame: np.ndarray,
        current_drawing_rect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> None:
        """
        Draw all defined ROIs on the frame.

        Args:
            frame: Frame to draw on (modified in-place)
            current_drawing_rect: Optional rect currently being drawn ((pt1, pt2))
        """
        # Draw existing ROIs
        for idx, (pt1, pt2) in enumerate(self.rois):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            thickness = (
                ROI_THICKNESS_SELECTED if idx == self.selected_roi_idx
                else ROI_THICKNESS_DEFAULT
            )
            cv2.rectangle(frame, pt1, pt2, color, thickness)

            # Draw index label near the top-left corner
            label = f"{idx + 1}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS
            )
            label_pos = (pt1[0] + 5, pt1[1] + text_height + 5)

            # Simple background for label visibility
            cv2.rectangle(
                frame,
                (pt1[0], pt1[1]),
                (pt1[0] + text_width + 10, pt1[1] + text_height + 10),
                (0, 0, 0),
                cv2.FILLED
            )
            cv2.putText(
                frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                ROI_LABEL_FONT_SCALE, color, ROI_LABEL_THICKNESS, cv2.LINE_AA
            )

        # Draw rectangle currently being drawn (in cyan)
        if current_drawing_rect is not None:
            pt1, pt2 = current_drawing_rect
            cv2.rectangle(frame, pt1, pt2, (0, 255, 255), ROI_THICKNESS_DEFAULT)

    def find_roi_at_point(self, point: Tuple[int, int]) -> Optional[int]:
        """
        Find ROI index at the given point.

        Args:
            point: (x, y) coordinates

        Returns:
            Index of ROI containing the point, or None if no ROI found
        """
        x, y = point
        for idx, (pt1, pt2) in enumerate(self.rois):
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def is_point_near_roi_corner(
        self,
        point: Tuple[int, int],
        roi_index: int,
        threshold: int = 10
    ) -> bool:
        """
        Check if point is near a corner of the specified ROI.

        Args:
            point: (x, y) coordinates
            roi_index: Index of ROI to check
            threshold: Distance threshold in pixels

        Returns:
            True if point is near a corner
        """
        if roi_index < 0 or roi_index >= len(self.rois):
            return False

        pt1, pt2 = self.rois[roi_index]
        x, y = point

        # Check all four corners
        corners = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        for cx, cy in corners:
            if abs(x - cx) <= threshold and abs(y - cy) <= threshold:
                return True

        return False

    def update_roi(
        self,
        index: int,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int]
    ) -> bool:
        """
        Update ROI coordinates.

        Args:
            index: ROI index to update
            pt1: New first corner (x, y)
            pt2: New second corner (x, y)

        Returns:
            True if successfully updated, False if index is invalid
        """
        if 0 <= index < len(self.rois):
            # Normalize points
            x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
            self.rois[index] = ((x1, y1), (x2, y2))
            return True
        return False

    def move_roi(self, index: int, dx: int, dy: int, frame_shape: Tuple[int, int]) -> bool:
        """
        Move an ROI by delta x and y.

        Args:
            index: ROI index to move
            dx: Horizontal displacement
            dy: Vertical displacement
            frame_shape: (height, width) to constrain movement

        Returns:
            True if successfully moved, False if index is invalid
        """
        if index < 0 or index >= len(self.rois):
            return False

        pt1, pt2 = self.rois[index]
        fh, fw = frame_shape

        # Calculate new positions
        new_x1 = max(0, min(pt1[0] + dx, fw - 1))
        new_y1 = max(0, min(pt1[1] + dy, fh - 1))
        new_x2 = max(0, min(pt2[0] + dx, fw - 1))
        new_y2 = max(0, min(pt2[1] + dy, fh - 1))

        self.rois[index] = ((new_x1, new_y1), (new_x2, new_y2))
        return True

    def resize_roi(
        self,
        index: int,
        new_pt2: Tuple[int, int],
        frame_shape: Tuple[int, int]
    ) -> bool:
        """
        Resize an ROI by updating the second corner point.

        Args:
            index: ROI index to resize
            new_pt2: New second corner (x, y)
            frame_shape: (height, width) to constrain resize

        Returns:
            True if successfully resized, False if index is invalid
        """
        if index < 0 or index >= len(self.rois):
            return False

        pt1, _ = self.rois[index]
        fh, fw = frame_shape

        # Constrain new point to frame boundaries
        new_x2 = max(0, min(new_pt2[0], fw - 1))
        new_y2 = max(0, min(new_pt2[1], fh - 1))

        # Ensure ROI has non-zero area
        if new_x2 != pt1[0] and new_y2 != pt1[1]:
            # Normalize coordinates
            x1, y1 = min(pt1[0], new_x2), min(pt1[1], new_y2)
            x2, y2 = max(pt1[0], new_x2), max(pt1[1], new_y2)
            self.rois[index] = ((x1, y1), (x2, y2))
            return True

        return False

    def extract_roi_from_frame(
        self,
        frame: np.ndarray,
        roi_index: int
    ) -> Optional[np.ndarray]:
        """
        Extract ROI pixels from frame.

        Args:
            frame: Video frame
            roi_index: Index of ROI to extract

        Returns:
            ROI as numpy array, or None if invalid
        """
        if roi_index < 0 or roi_index >= len(self.rois):
            return None

        pt1, pt2 = self.rois[roi_index]
        roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        if roi.size == 0:
            return None

        return roi

    def get_non_background_roi_indices(self) -> List[int]:
        """
        Get list of all ROI indices except the background ROI.

        Returns:
            List of ROI indices (excluding background)
        """
        indices = []
        for idx in range(len(self.rois)):
            if idx != self.background_roi_idx:
                indices.append(idx)
        return indices