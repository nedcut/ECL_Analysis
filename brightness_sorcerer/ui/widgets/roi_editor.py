"""
ROI Editor Widget

Modern ROI creation and editing widget with enhanced visualization.
"""

from typing import List, Tuple, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


class ROIEditorWidget(QtWidgets.QWidget):
    """
    Enhanced ROI editor with modern visual design and improved interaction.
    """
    
    # Signals
    roiCreated = QtCore.pyqtSignal(int, tuple, tuple)  # ROI index, pt1, pt2
    roiDeleted = QtCore.pyqtSignal(int)                # ROI index
    roiSelected = QtCore.pyqtSignal(int)               # ROI index
    backgroundRoiSet = QtCore.pyqtSignal(int)          # ROI index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rois: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.selected_roi = None
        self.background_roi = None
        
        self._setup_ui()
        self._setup_styles()
    
    def _setup_ui(self):
        """Create and layout UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("ROI Editor")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        # ROI list
        self.roi_list = QtWidgets.QListWidget()
        layout.addWidget(self.roi_list, 1)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.add_button = QtWidgets.QPushButton("Add ROI")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.set_background_button = QtWidgets.QPushButton("Set as Background")
        
        controls_layout.addWidget(self.add_button)
        controls_layout.addWidget(self.delete_button)
        controls_layout.addWidget(self.set_background_button)
        
        layout.addLayout(controls_layout)
    
    def _setup_styles(self):
        """Apply modern styling."""
        self.setStyleSheet("""
            ROIEditorWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #3daee9;
            }
            QPushButton {
                background-color: #3daee9;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #5cbef4;
            }
        """)
    
    def add_roi(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> int:
        """Add a new ROI."""
        roi_index = len(self.rois)
        self.rois.append((pt1, pt2))
        
        item = QtWidgets.QListWidgetItem(f"ROI {roi_index + 1}")
        self.roi_list.addItem(item)
        
        self.roiCreated.emit(roi_index, pt1, pt2)
        return roi_index
    
    def delete_roi(self, index: int):
        """Delete an ROI."""
        if 0 <= index < len(self.rois):
            self.rois.pop(index)
            self.roi_list.takeItem(index)
            self.roiDeleted.emit(index)
    
    def select_roi(self, index: int):
        """Select an ROI."""
        if 0 <= index < len(self.rois):
            self.selected_roi = index
            self.roi_list.setCurrentRow(index)
            self.roiSelected.emit(index)
    
    def set_background_roi(self, index: int):
        """Set an ROI as background."""
        if 0 <= index < len(self.rois):
            self.background_roi = index
            self.backgroundRoiSet.emit(index)