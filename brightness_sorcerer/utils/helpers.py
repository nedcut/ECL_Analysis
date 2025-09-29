"""Utility helper functions for common operations."""

from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets


def normalize_roi_bounds(
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Normalize ROI bounds to ensure they are within frame boundaries.

    Args:
        pt1: First corner point (x, y)
        pt2: Second corner point (x, y)
        frame_shape: Shape of the frame (height, width, channels)

    Returns:
        Tuple of (x1, y1, x2, y2) normalized to frame boundaries
    """
    fh, fw = frame_shape[:2]
    x1 = max(0, min(pt1[0], fw - 1))
    y1 = max(0, min(pt1[1], fh - 1))
    x2 = max(0, min(pt2[0], fw - 1))
    y2 = max(0, min(pt2[1], fh - 1))
    return x1, y1, x2, y2


def convert_to_lab_lstar(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert BGR ROI to CIE LAB L* channel (0-100 scale).

    Args:
        roi_bgr: ROI in BGR color space

    Returns:
        L* channel array (0-100 scale) or None if conversion fails
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0
        return l_star
    except Exception:
        return None


def create_progress_dialog(
    title: str,
    label: str,
    max_value: int,
    parent: Optional[QtWidgets.QWidget] = None
) -> QtWidgets.QProgressDialog:
    """
    Create a standard progress dialog with consistent styling.

    Args:
        title: Dialog window title
        label: Label text describing the operation
        max_value: Maximum progress value
        parent: Parent widget (optional)

    Returns:
        Configured QProgressDialog
    """
    progress = QtWidgets.QProgressDialog(label, "Cancel", 0, max_value, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    return progress


def apply_morphological_filter(
    mask: np.ndarray,
    kernel_size: int = 3,
    operation: int = cv2.MORPH_OPEN
) -> np.ndarray:
    """
    Apply morphological filtering to clean up a binary mask.

    Args:
        mask: Boolean or uint8 binary mask
        kernel_size: Size of the morphological kernel
        operation: Morphological operation (e.g., cv2.MORPH_OPEN)

    Returns:
        Cleaned binary mask
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert boolean mask to uint8 if necessary
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask

    # Apply morphological operation
    cleaned_mask = cv2.morphologyEx(mask_uint8, operation, kernel)

    # Convert back to boolean if input was boolean
    if mask.dtype == bool:
        return cleaned_mask > 0
    else:
        return cleaned_mask


def extract_blue_channel(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract blue channel from BGR image.

    Args:
        roi_bgr: ROI in BGR color space

    Returns:
        Blue channel array (0-255 scale) or None if extraction fails
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    try:
        # Blue is channel 0 in BGR format
        blue_chan = roi_bgr[:, :, 0].astype(np.float32)
        return blue_chan
    except Exception:
        return None


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm timestamp.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def validate_roi(x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Validate that ROI coordinates form a valid rectangle.

    Args:
        x1, y1: First corner coordinates
        x2, y2: Second corner coordinates

    Returns:
        True if ROI is valid (has non-zero area)
    """
    return abs(x2 - x1) > 0 and abs(y2 - y1) > 0