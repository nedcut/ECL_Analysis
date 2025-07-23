"""
Main Toolbar Widget  

Modern application toolbar with enhanced icons and improved layout.
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class MainToolbar(QtWidgets.QToolBar):
    """
    Enhanced main toolbar with modern design and improved functionality.
    """
    
    # Signals
    openVideoRequested = QtCore.pyqtSignal()
    saveProjectRequested = QtCore.pyqtSignal()
    loadProjectRequested = QtCore.pyqtSignal()
    settingsRequested = QtCore.pyqtSignal()
    helpRequested = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mainToolbar")
        self._setup_toolbar()
        self._setup_styles()
    
    def _setup_toolbar(self):
        """Create toolbar actions and layout."""
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.setIconSize(QtCore.QSize(24, 24))
        
        # File operations
        self.open_action = self.addAction("Open Video")
        self.open_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.open_action.setToolTip("Open video file for analysis")
        
        self.addSeparator()
        
        self.save_action = self.addAction("Save Project")
        self.save_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_action.setToolTip("Save current project")
        
        self.load_action = self.addAction("Load Project")  
        self.load_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.load_action.setToolTip("Load existing project")
        
        self.addSeparator()
        
        # Settings and help
        self.settings_action = self.addAction("Settings")
        self.settings_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.settings_action.setToolTip("Application settings")
        
        self.help_action = self.addAction("Help")
        self.help_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogHelpButton))
        self.help_action.setToolTip("Help and documentation")
        
        # Connect signals
        self.open_action.triggered.connect(self.openVideoRequested.emit)
        self.save_action.triggered.connect(self.saveProjectRequested.emit)
        self.load_action.triggered.connect(self.loadProjectRequested.emit)
        self.settings_action.triggered.connect(self.settingsRequested.emit)
        self.help_action.triggered.connect(self.helpRequested.emit)
    
    def _setup_styles(self):
        """Apply modern styling."""
        self.setStyleSheet("""
            QToolBar#mainToolbar {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                spacing: 4px;
                color: #ffffff;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                min-width: 60px;
            }
            QToolButton:hover {
                background-color: #3daee9;
            }
            QToolButton:pressed {
                background-color: #2a9dd9;
            }
        """)