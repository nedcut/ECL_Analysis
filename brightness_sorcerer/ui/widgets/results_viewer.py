"""
Results Viewer Widget

Modern results visualization widget with enhanced charts and data display.
"""

from typing import Dict, List, Any
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


class ResultsViewerWidget(QtWidgets.QWidget):
    """
    Enhanced results viewer with modern charts and improved data presentation.
    """
    
    # Signals
    exportRequested = QtCore.pyqtSignal(str)  # Export format
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data: Dict[str, Any] = {}
        
        self._setup_ui()
        self._setup_styles()
    
    def _setup_ui(self):
        """Create and layout UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("Analysis Results")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        # Results display area
        self.results_area = QtWidgets.QTextEdit()
        self.results_area.setReadOnly(True)
        layout.addWidget(self.results_area, 1)
        
        # Export controls
        export_layout = QtWidgets.QHBoxLayout()
        
        self.export_button = QtWidgets.QPushButton("Export Results")
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["CSV", "JSON", "Excel"])
        
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.format_combo)
        export_layout.addStretch()
        
        layout.addLayout(export_layout)
    
    def _setup_styles(self):
        """Apply modern styling."""
        self.setStyleSheet("""
            ResultsViewerWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #3daee9;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                font-family: "Courier New", monospace;
            }
            QPushButton {
                background-color: #3daee9;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                color: white;
            }
        """)
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results."""
        self.results_data = results
        
        # Format results for display
        text = "Analysis Results\n" + "="*50 + "\n\n"
        
        for key, value in results.items():
            text += f"{key}: {value}\n"
        
        self.results_area.setPlainText(text)
    
    def clear_results(self):
        """Clear displayed results."""
        self.results_data = {}
        self.results_area.clear()