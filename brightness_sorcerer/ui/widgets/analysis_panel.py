"""
Analysis Panel Widget

Modern analysis control panel with enhanced UI/UX.
"""

from typing import Optional, Dict, Any
from PyQt5 import QtCore, QtGui, QtWidgets


class AnalysisPanelWidget(QtWidgets.QWidget):
    """
    Enhanced analysis control panel with modern design and improved usability.
    """
    
    # Signals
    analysisRequested = QtCore.pyqtSignal(dict)  # Analysis parameters
    settingsChanged = QtCore.pyqtSignal(dict)    # Settings changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings: Dict[str, Any] = {}
        
        self._setup_ui()
        self._setup_styles() 
        self._connect_signals()
        self._load_default_settings()
    
    def _setup_ui(self):
        """Create and layout UI components."""
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Title
        title_label = QtWidgets.QLabel("Analysis Settings")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Settings scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarNever)
        
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(12)
        
        # Analysis Method Group
        method_group = self._create_analysis_method_group()
        settings_layout.addWidget(method_group)
        
        # Brightness Settings Group
        brightness_group = self._create_brightness_settings_group()
        settings_layout.addWidget(brightness_group)
        
        # Advanced Settings Group
        advanced_group = self._create_advanced_settings_group()
        settings_layout.addWidget(advanced_group)
        
        # Export Settings Group
        export_group = self._create_export_settings_group()
        settings_layout.addWidget(export_group)
        
        settings_layout.addStretch()
        
        scroll_area.setWidget(settings_widget)
        layout.addWidget(scroll_area, 1)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.setObjectName("secondaryButton")
        
        self.analyze_button = QtWidgets.QPushButton("Start Analysis")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setEnabled(False)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.analyze_button)
        
        layout.addLayout(button_layout)
    
    def _create_analysis_method_group(self) -> QtWidgets.QGroupBox:
        """Create analysis method selection group."""
        group = QtWidgets.QGroupBox("Analysis Method")
        layout = QtWidgets.QVBoxLayout(group)
        
        self.method_basic = QtWidgets.QRadioButton("Basic (L* channel only)")
        self.method_enhanced = QtWidgets.QRadioButton("Enhanced (L* + Blue channel)")
        self.method_enhanced.setChecked(True)
        
        self.method_basic.setToolTip("Standard CIE LAB L* channel analysis")
        self.method_enhanced.setToolTip("Enhanced analysis with blue channel for bioluminescence")
        
        layout.addWidget(self.method_basic)
        layout.addWidget(self.method_enhanced)
        
        return group
    
    def _create_brightness_settings_group(self) -> QtWidgets.QGroupBox:
        """Create brightness analysis settings group."""
        group = QtWidgets.QGroupBox("Brightness Analysis")
        layout = QtWidgets.QFormLayout(group)
        
        # Background subtraction
        self.background_checkbox = QtWidgets.QCheckBox("Enable background subtraction")
        self.background_checkbox.setChecked(True)
        layout.addRow(self.background_checkbox)
        
        # Threshold value
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setValue(10.0)
        self.threshold_spin.setSuffix(" L*")
        self.threshold_spin.setToolTip("Brightness threshold for analysis")
        layout.addRow("Threshold:", self.threshold_spin)
        
        # Morphological cleanup
        self.morphology_checkbox = QtWidgets.QCheckBox("Morphological cleanup")
        self.morphology_checkbox.setChecked(True)
        self.morphology_checkbox.setToolTip("Remove noise using morphological operations")
        layout.addRow(self.morphology_checkbox)
        
        return group
    
    def _create_advanced_settings_group(self) -> QtWidgets.QGroupBox:
        """Create advanced settings group."""
        group = QtWidgets.QGroupBox("Advanced Settings")
        group.setCheckable(True)
        group.setChecked(False)
        
        layout = QtWidgets.QFormLayout(group)
        
        # Gaussian blur
        self.blur_spin = QtWidgets.QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 5.0)
        self.blur_spin.setValue(0.0)
        self.blur_spin.setSingleStep(0.1)
        self.blur_spin.setToolTip("Gaussian blur sigma (0 = disabled)")
        layout.addRow("Gaussian Blur σ:", self.blur_spin)
        
        # Low-light enhancement
        self.enhancement_checkbox = QtWidgets.QCheckBox("Low-light enhancement")
        self.enhancement_checkbox.setToolTip("Enhanced processing for low-light conditions")
        layout.addRow(self.enhancement_checkbox)
        
        # Reference mask usage
        self.reference_mask_checkbox = QtWidgets.QCheckBox("Use reference masks")
        self.reference_mask_checkbox.setChecked(True)
        self.reference_mask_checkbox.setToolTip("Use reference masks for consistent analysis")
        layout.addRow(self.reference_mask_checkbox)
        
        return group
    
    def _create_export_settings_group(self) -> QtWidgets.QGroupBox:
        """Create export settings group."""
        group = QtWidgets.QGroupBox("Export Options")
        layout = QtWidgets.QVBoxLayout(group)
        
        # Export format
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.addWidget(QtWidgets.QLabel("Format:"))
        
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["CSV", "JSON", "Excel", "HDF5"])
        self.format_combo.setCurrentText("CSV")
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        layout.addLayout(format_layout)
        
        # Export options
        self.export_plots_checkbox = QtWidgets.QCheckBox("Generate plots")
        self.export_plots_checkbox.setChecked(True)
        
        self.export_summary_checkbox = QtWidgets.QCheckBox("Include summary statistics")
        self.export_summary_checkbox.setChecked(True)
        
        layout.addWidget(self.export_plots_checkbox)
        layout.addWidget(self.export_summary_checkbox)
        
        return group
    
    def _setup_styles(self):
        """Apply modern styling to the widget."""
        self.setStyleSheet("""
            AnalysisPanelWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            QLabel#titleLabel {
                font-size: 16px;
                font-weight: bold;
                color: #3daee9;
                margin-bottom: 8px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3daee9;
            }
            
            QPushButton#primaryButton {
                background-color: #3daee9;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                color: white;
                min-width: 120px;
            }
            
            QPushButton#primaryButton:hover {
                background-color: #5cbef4;
            }
            
            QPushButton#primaryButton:pressed {
                background-color: #2a9dd9;
            }
            
            QPushButton#primaryButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            
            QPushButton#secondaryButton {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 10px 20px;
                color: white;
                min-width: 120px;
            }
            
            QPushButton#secondaryButton:hover {
                background-color: #3b3b3b;
                border-color: #777777;
            }
            
            QRadioButton {
                spacing: 5px;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            
            QRadioButton::indicator:unchecked {
                border: 2px solid #555555;
                border-radius: 8px;
                background-color: #2b2b2b;
            }
            
            QRadioButton::indicator:checked {
                border: 2px solid #3daee9;
                border-radius: 8px;
                background-color: #3daee9;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #3daee9;
                border-radius: 3px;
                background-color: #3daee9;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
                min-width: 60px;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #3daee9;
            }
            
            QComboBox {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
                min-width: 80px;
            }
            
            QComboBox:focus {
                border-color: #3daee9;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border: none;
                background-color: #3daee9;
                width: 8px;
                height: 8px;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
        """)
    
    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        self.reset_button.clicked.connect(self._reset_to_defaults)
        
        # Connect settings change signals
        self.method_basic.toggled.connect(self._on_settings_changed)
        self.method_enhanced.toggled.connect(self._on_settings_changed)
        self.background_checkbox.toggled.connect(self._on_settings_changed)
        self.threshold_spin.valueChanged.connect(self._on_settings_changed)
        self.morphology_checkbox.toggled.connect(self._on_settings_changed)
        self.blur_spin.valueChanged.connect(self._on_settings_changed)
        self.enhancement_checkbox.toggled.connect(self._on_settings_changed)
        self.reference_mask_checkbox.toggled.connect(self._on_settings_changed)
        self.format_combo.currentTextChanged.connect(self._on_settings_changed)
        self.export_plots_checkbox.toggled.connect(self._on_settings_changed)
        self.export_summary_checkbox.toggled.connect(self._on_settings_changed)
    
    def _load_default_settings(self):
        """Load default analysis settings."""
        self.settings = {
            'analysis_method': 'enhanced',
            'background_subtraction': True,
            'threshold': 10.0,
            'morphological_cleanup': True,
            'gaussian_blur_sigma': 0.0,
            'low_light_enhancement': False,
            'use_reference_masks': True,
            'export_format': 'CSV',
            'generate_plots': True,
            'include_summary': True
        }
        self._update_ui_from_settings()
    
    def _update_ui_from_settings(self):
        """Update UI controls from current settings."""
        self.method_enhanced.setChecked(self.settings['analysis_method'] == 'enhanced')
        self.method_basic.setChecked(self.settings['analysis_method'] == 'basic')
        self.background_checkbox.setChecked(self.settings['background_subtraction'])
        self.threshold_spin.setValue(self.settings['threshold'])
        self.morphology_checkbox.setChecked(self.settings['morphological_cleanup'])
        self.blur_spin.setValue(self.settings['gaussian_blur_sigma'])
        self.enhancement_checkbox.setChecked(self.settings['low_light_enhancement'])
        self.reference_mask_checkbox.setChecked(self.settings['use_reference_masks'])
        self.format_combo.setCurrentText(self.settings['export_format'])
        self.export_plots_checkbox.setChecked(self.settings['generate_plots'])
        self.export_summary_checkbox.setChecked(self.settings['include_summary'])
    
    def _get_settings_from_ui(self) -> Dict[str, Any]:
        """Get current settings from UI controls."""
        return {
            'analysis_method': 'enhanced' if self.method_enhanced.isChecked() else 'basic',
            'background_subtraction': self.background_checkbox.isChecked(),
            'threshold': self.threshold_spin.value(),
            'morphological_cleanup': self.morphology_checkbox.isChecked(),
            'gaussian_blur_sigma': self.blur_spin.value(),
            'low_light_enhancement': self.enhancement_checkbox.isChecked(),
            'use_reference_masks': self.reference_mask_checkbox.isChecked(),
            'export_format': self.format_combo.currentText(),
            'generate_plots': self.export_plots_checkbox.isChecked(),
            'include_summary': self.export_summary_checkbox.isChecked()
        }
    
    def _on_settings_changed(self):
        """Handle settings change."""
        self.settings = self._get_settings_from_ui()
        self.settingsChanged.emit(self.settings)
    
    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        settings = self._get_settings_from_ui()
        self.analysisRequested.emit(settings)
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        self._load_default_settings()
        self.settingsChanged.emit(self.settings)
    
    def set_analysis_enabled(self, enabled: bool):
        """Enable or disable the analysis button."""
        self.analyze_button.setEnabled(enabled)
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get the current analysis settings."""
        return self.settings.copy()
    
    def load_settings(self, settings: Dict[str, Any]):
        """Load settings from a dictionary."""
        self.settings.update(settings)
        self._update_ui_from_settings()
        self.settingsChanged.emit(self.settings)