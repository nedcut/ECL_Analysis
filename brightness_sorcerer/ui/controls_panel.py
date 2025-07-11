"""Controls panel widget for Brightness Sorcerer."""

from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional, Callable, List
import os
import logging

from ..core.video_processor import VideoProcessor
from ..core.roi_manager import ROIManager
from ..core.brightness_analyzer import BrightnessAnalyzer
from ..core.settings_manager import SettingsManager
from ..models.roi import ROI


class ROIListWidget(QtWidgets.QWidget):
    """Widget for managing ROI list."""
    
    # Signals
    roi_selected = QtCore.pyqtSignal(int)  # roi_index
    roi_deleted = QtCore.pyqtSignal(int)   # roi_index
    background_roi_changed = QtCore.pyqtSignal(int)  # roi_index or -1 for none
    
    def __init__(self, app_controller, parent=None):
        super().__init__(parent)
        self.app_controller = app_controller
        self._setup_ui()
        self._connect_signals()
        
        # Register UI callbacks
        self.app_controller.register_ui_callback('roi_added', self._update_roi_list)
        self.app_controller.register_ui_callback('roi_deleted', self._update_roi_list)
        self.app_controller.register_ui_callback('roi_selected', self._update_selection)
        self.app_controller.register_ui_callback('roi_renamed', self._update_roi_list)
        self.app_controller.register_ui_callback('background_roi_changed', self._update_roi_list)
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Header
        header_layout = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel("Regions of Interest")
        header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(header_label)
        
        # Add ROI button
        self.add_roi_btn = QtWidgets.QPushButton("Add ROI")
        self.add_roi_btn.setMaximumWidth(80)
        header_layout.addWidget(self.add_roi_btn)
        
        layout.addLayout(header_layout)
        
        # ROI list
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(150)
        self.roi_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        layout.addWidget(self.roi_list)
        
        # ROI controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.delete_roi_btn = QtWidgets.QPushButton("Delete")
        self.delete_roi_btn.setEnabled(False)
        self.delete_roi_btn.setProperty("secondary", True)
        controls_layout.addWidget(self.delete_roi_btn)
        
        self.set_background_btn = QtWidgets.QPushButton("Set as Background")
        self.set_background_btn.setEnabled(False)
        self.set_background_btn.setProperty("secondary", True)
        controls_layout.addWidget(self.set_background_btn)
        
        layout.addLayout(controls_layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.add_roi_btn.clicked.connect(self._on_add_roi_clicked)
        self.delete_roi_btn.clicked.connect(self._on_delete_roi_clicked)
        self.set_background_btn.clicked.connect(self._on_set_background_clicked)
        self.roi_list.itemSelectionChanged.connect(self._on_roi_selection_changed)
        self.roi_list.customContextMenuRequested.connect(self._show_context_menu)
    
    def _update_roi_list(self, *args):
        """Update the ROI list display."""
        self.roi_list.clear()
        
        for i, roi in enumerate(self.app_controller.roi_manager.rois):
            item_text = roi.label
            if roi.is_background:
                item_text += " (Background)"
            
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)  # Store ROI index
            
            # Color indicator
            color = QtGui.QColor(*roi.color)
            color_pixmap = QtGui.QPixmap(16, 16)
            color_pixmap.fill(color)
            item.setIcon(QtGui.QIcon(color_pixmap))
            
            self.roi_list.addItem(item)
    
    def _update_selection(self, roi_index=None):
        """Update selection in the list."""
        self.roi_list.blockSignals(True)
        
        selected_index = self.app_controller.roi_manager.selected_roi_index
        if selected_index is not None:
            self.roi_list.setCurrentRow(selected_index)
            self.delete_roi_btn.setEnabled(True)
            self.set_background_btn.setEnabled(True)
        else:
            self.roi_list.clearSelection()
            self.delete_roi_btn.setEnabled(False)
            self.set_background_btn.setEnabled(False)
        
        self.roi_list.blockSignals(False)
    
    def _on_add_roi_clicked(self):
        """Handle add ROI button click."""
        # Add default ROI in center of frame
        result = self.app_controller.add_roi(100, 100, 200, 150, "ROI")
        if result.is_error():
            logging.error(f"Failed to add ROI: {result.error}")
    
    def _on_delete_roi_clicked(self):
        """Handle delete ROI button click."""
        selected_index = self.app_controller.roi_manager.selected_roi_index
        if selected_index is not None:
            result = self.app_controller.delete_roi(selected_index)
            if result.is_success():
                self.roi_deleted.emit(selected_index)
            else:
                logging.error(f"Failed to delete ROI: {result.error}")
    
    def _on_set_background_clicked(self):
        """Handle set background button click."""
        selected_index = self.app_controller.roi_manager.selected_roi_index
        if selected_index is not None:
            current_bg = self.app_controller.roi_manager.background_roi_index
            
            if current_bg == selected_index:
                # Unset background
                result = self.app_controller.set_background_roi(None)
                if result.is_success():
                    self.background_roi_changed.emit(-1)
            else:
                # Set as background
                result = self.app_controller.set_background_roi(selected_index)
                if result.is_success():
                    self.background_roi_changed.emit(selected_index)
            
            if result.is_error():
                logging.error(f"Failed to set background ROI: {result.error}")
    
    def _on_roi_selection_changed(self):
        """Handle ROI selection change in list."""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_index = current_item.data(QtCore.Qt.UserRole)
            self.app_controller.select_roi(roi_index)
            self.roi_selected.emit(roi_index)
        else:
            self.app_controller.select_roi(None)
    
    def _show_context_menu(self, position):
        """Show context menu for ROI list."""
        item = self.roi_list.itemAt(position)
        if item is None:
            return
        
        roi_index = item.data(QtCore.Qt.UserRole)
        roi = self.app_controller.roi_manager.get_roi(roi_index)
        if roi is None:
            return
        
        menu = QtWidgets.QMenu(self)
        
        # Rename action
        rename_action = menu.addAction("Rename...")
        rename_action.triggered.connect(lambda: self._rename_roi(roi_index))
        
        # Background toggle
        if roi.is_background:
            bg_action = menu.addAction("Remove as Background")
            bg_action.triggered.connect(lambda: self._toggle_background(roi_index))
        else:
            bg_action = menu.addAction("Set as Background")
            bg_action.triggered.connect(lambda: self._toggle_background(roi_index))
        
        menu.addSeparator()
        
        # Delete action
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_roi(roi_index))
        
        menu.exec_(self.roi_list.mapToGlobal(position))
    
    def _rename_roi(self, roi_index: int):
        """Rename an ROI."""
        roi = self.app_controller.roi_manager.get_roi(roi_index)
        if roi is None:
            return
        
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename ROI", "Enter new name:", text=roi.label
        )
        
        if ok and new_name.strip():
            result = self.app_controller.rename_roi(roi_index, new_name.strip())
            if result.is_error():
                logging.error(f"Failed to rename ROI: {result.error}")
    
    def _toggle_background(self, roi_index: int):
        """Toggle background status of ROI."""
        roi = self.app_controller.roi_manager.get_roi(roi_index)
        if roi is None:
            return
        
        if roi.is_background:
            result = self.app_controller.set_background_roi(None)
            if result.is_success():
                self.background_roi_changed.emit(-1)
        else:
            result = self.app_controller.set_background_roi(roi_index)
            if result.is_success():
                self.background_roi_changed.emit(roi_index)
        
        if result.is_error():
            logging.error(f"Failed to toggle background ROI: {result.error}")
    
    def _delete_roi(self, roi_index: int):
        """Delete an ROI."""
        result = self.app_controller.delete_roi(roi_index)
        if result.is_success():
            self.roi_deleted.emit(roi_index)
        else:
            logging.error(f"Failed to delete ROI: {result.error}")


class AnalysisControlsWidget(QtWidgets.QWidget):
    """Widget for analysis controls."""
    
    # Signals
    analyze_requested = QtCore.pyqtSignal(int, int, str)  # start_frame, end_frame, output_dir
    auto_detect_requested = QtCore.pyqtSignal()
    
    def __init__(self, app_controller, parent=None):
        super().__init__(parent)
        self.app_controller = app_controller
        self._setup_ui()
        self._connect_signals()
        self._load_settings()
        
        # Register UI callbacks
        self.app_controller.register_ui_callback('video_loaded', self._on_video_loaded)
        self.app_controller.register_ui_callback('analysis_progress', self.show_progress)
        self.app_controller.register_ui_callback('analysis_completed', self._on_analysis_completed)
        self.app_controller.register_ui_callback('analysis_failed', self._on_analysis_failed)
        self.app_controller.register_ui_callback('frame_range_detected', self.set_auto_detected_range)
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Analysis range group
        range_group = QtWidgets.QGroupBox("Analysis Range")
        range_layout = QtWidgets.QFormLayout(range_group)
        
        self.start_frame_spin = QtWidgets.QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setEnabled(False)
        range_layout.addRow("Start Frame:", self.start_frame_spin)
        
        self.end_frame_spin = QtWidgets.QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setEnabled(False)
        range_layout.addRow("End Frame:", self.end_frame_spin)
        
        # Range buttons
        range_btn_layout = QtWidgets.QHBoxLayout()
        
        self.set_start_btn = QtWidgets.QPushButton("Set Start")
        self.set_start_btn.setEnabled(False)
        self.set_start_btn.setProperty("secondary", True)
        range_btn_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QtWidgets.QPushButton("Set End")
        self.set_end_btn.setEnabled(False)
        self.set_end_btn.setProperty("secondary", True)
        range_btn_layout.addWidget(self.set_end_btn)
        
        range_layout.addRow(range_btn_layout)
        
        self.auto_detect_btn = QtWidgets.QPushButton("Auto-Detect Range")
        self.auto_detect_btn.setEnabled(False)
        range_layout.addRow(self.auto_detect_btn)
        
        layout.addWidget(range_group)
        
        # Analysis parameters group
        params_group = QtWidgets.QGroupBox("Analysis Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setSuffix(" L* units")
        params_layout.addRow("Manual Threshold:", self.threshold_spin)
        
        self.noise_floor_spin = QtWidgets.QDoubleSpinBox()
        self.noise_floor_spin.setRange(0.0, 50.0)
        self.noise_floor_spin.setSingleStep(1.0)
        self.noise_floor_spin.setSuffix(" L* units")
        params_layout.addRow("Noise Floor:", self.noise_floor_spin)
        
        layout.addWidget(params_group)
        
        # Output settings group
        output_group = QtWidgets.QGroupBox("Output Settings")
        output_layout = QtWidgets.QFormLayout(output_group)
        
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        output_layout.addRow("Output Directory:", self.output_dir_edit)
        
        self.browse_output_btn = QtWidgets.QPushButton("Browse...")
        self.browse_output_btn.setMaximumWidth(80)
        self.browse_output_btn.setProperty("secondary", True)
        output_layout.addRow("", self.browse_output_btn)
        
        layout.addWidget(output_group)
        
        # Analysis button
        self.analyze_btn = QtWidgets.QPushButton("Analyze Brightness")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        layout.addStretch()
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.set_start_btn.clicked.connect(self._set_start_frame)
        self.set_end_btn.clicked.connect(self._set_end_frame)
        self.auto_detect_btn.clicked.connect(self._auto_detect_range)
        self.browse_output_btn.clicked.connect(self._browse_output_directory)
        self.analyze_btn.clicked.connect(self._start_analysis)
        
        # Parameter changes
        self.threshold_spin.valueChanged.connect(self._save_settings)
        self.noise_floor_spin.valueChanged.connect(self._save_settings)
    
    def _load_settings(self):
        """Load settings from settings manager."""
        defaults = self.app_controller.get_analysis_defaults()
        self.threshold_spin.setValue(defaults['manual_threshold'])
        self.noise_floor_spin.setValue(defaults['noise_floor'])
    
    def _save_settings(self):
        """Save current settings."""
        result = self.app_controller.update_threshold_settings(
            manual_threshold=self.threshold_spin.value(),
            noise_floor=self.noise_floor_spin.value()
        )
        if result.is_error():
            logging.error(f"Failed to save threshold settings: {result.error}")
    
    def _on_video_loaded(self, video_data):
        """Handle video loaded from controller."""
        total_frames = video_data.total_frames
        
        # Update frame range controls
        self.start_frame_spin.setMaximum(total_frames - 1)
        self.end_frame_spin.setMaximum(total_frames - 1)
        self.start_frame_spin.setValue(0)
        self.end_frame_spin.setValue(total_frames - 1)
        
        # Enable controls
        self.start_frame_spin.setEnabled(True)
        self.end_frame_spin.setEnabled(True)
        self.set_start_btn.setEnabled(True)
        self.set_end_btn.setEnabled(True)
        self.auto_detect_btn.setEnabled(True)
        
        self._update_analyze_button()
    
    def update_video_loaded(self, loaded: bool):
        """Update controls when video is loaded/unloaded (legacy method)."""
        if not loaded:
            # Disable controls
            self.start_frame_spin.setEnabled(False)
            self.end_frame_spin.setEnabled(False)
            self.set_start_btn.setEnabled(False)
            self.set_end_btn.setEnabled(False)
            self.auto_detect_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
    
    def _set_start_frame(self):
        """Set start frame to current frame."""
        if self.app_controller.video_processor.is_loaded():
            current_frame = self.app_controller.video_processor.current_frame_index
            self.start_frame_spin.setValue(current_frame)
    
    def _set_end_frame(self):
        """Set end frame to current frame."""
        if self.app_controller.video_processor.is_loaded():
            current_frame = self.app_controller.video_processor.current_frame_index
            self.end_frame_spin.setValue(current_frame)
    
    def _auto_detect_range(self):
        """Request auto-detection of frame range."""
        result = self.app_controller.auto_detect_frame_range()
        if result.is_error():
            QtWidgets.QMessageBox.warning(
                self, "Auto-Detection Failed",
                f"Could not auto-detect frame range: {result.error}"
            )
    
    def _browse_output_directory(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)
            self._update_analyze_button()
    
    def _update_analyze_button(self):
        """Update analyze button enabled state."""
        video_loaded = self.app_controller.video_processor.is_loaded()
        output_selected = bool(self.output_dir_edit.text().strip())
        
        self.analyze_btn.setEnabled(video_loaded and output_selected)
    
    def _start_analysis(self):
        """Start brightness analysis."""
        if not self.app_controller.video_processor.is_loaded():
            return
        
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        output_dir = self.output_dir_edit.text().strip()
        
        if not output_dir or not os.path.exists(output_dir):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Output Directory",
                "Please select a valid output directory."
            )
            return
        
        if start_frame >= end_frame:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Frame Range",
                "Start frame must be less than end frame."
            )
            return
        
        # Start analysis via controller
        result = self.app_controller.start_analysis(start_frame, end_frame, output_dir)
        if result.is_error():
            QtWidgets.QMessageBox.critical(
                self, "Analysis Failed",
                f"Could not start analysis: {result.error}"
            )
    
    def show_progress(self, current: int, total: int, message: str = ""):
        """Show analysis progress."""
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
        self.analyze_btn.setEnabled(False)
    
    def hide_progress(self):
        """Hide analysis progress."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._update_analyze_button()
    
    def set_auto_detected_range(self, start_frame: int, end_frame: int):
        """Set auto-detected frame range."""
        self.start_frame_spin.setValue(start_frame)
        self.end_frame_spin.setValue(end_frame)
    
    def _on_analysis_completed(self, success: bool):
        """Handle analysis completion."""
        self.hide_progress()
        if success:
            QtWidgets.QMessageBox.information(
                self, "Analysis Complete",
                "Brightness analysis completed successfully!"
            )
    
    def _on_analysis_failed(self, error_message: str):
        """Handle analysis failure."""
        self.hide_progress()
        QtWidgets.QMessageBox.critical(
            self, "Analysis Failed",
            f"Analysis failed: {error_message}"
        )


class ControlsPanel(QtWidgets.QWidget):
    """Main controls panel containing all control widgets."""
    
    def __init__(self, app_controller, parent=None):
        super().__init__(parent)
        self.app_controller = app_controller
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the controls panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # ROI list widget
        self.roi_widget = ROIListWidget(self.app_controller)
        layout.addWidget(self.roi_widget)
        
        # Analysis controls widget
        self.analysis_widget = AnalysisControlsWidget(self.app_controller)
        layout.addWidget(self.analysis_widget)
        
        # Set minimum width
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
    
    def _connect_signals(self):
        """Connect panel signals."""
        # Forward signals from child widgets
        self.roi_widget.roi_selected.connect(self._on_roi_selected)
        self.roi_widget.roi_deleted.connect(self._on_roi_deleted)
        self.roi_widget.background_roi_changed.connect(self._on_background_roi_changed)
        
        self.analysis_widget.analyze_requested.connect(self._on_analyze_requested)
        self.analysis_widget.auto_detect_requested.connect(self._on_auto_detect_requested)
    
    def update_video_loaded(self, loaded: bool):
        """Update panel when video is loaded/unloaded."""
        self.analysis_widget.update_video_loaded(loaded)
    
    def _on_roi_selected(self, roi_index: int):
        """Handle ROI selection."""
        # This is handled by the ROI manager already
        pass
    
    def _on_roi_deleted(self, roi_index: int):
        """Handle ROI deletion."""
        # This is handled by the ROI manager already
        pass
    
    def _on_background_roi_changed(self, roi_index: int):
        """Handle background ROI change."""
        # This is handled by the ROI manager already
        pass
    
    def _on_analyze_requested(self, start_frame: int, end_frame: int, output_dir: str):
        """Handle analysis request."""
        # This should be connected to the main window
        pass
    
    def _on_auto_detect_requested(self):
        """Handle auto-detect request."""
        # This should be connected to the main window
        pass