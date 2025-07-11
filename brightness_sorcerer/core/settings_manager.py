"""Settings and configuration management for Brightness Sorcerer."""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Constants
DEFAULT_SETTINGS_FILE = "brightness_analyzer_settings.json"
MAX_RECENT_FILES = 10


@dataclass
class AppSettings:
    """Application settings data structure."""
    
    # Window settings
    window_geometry: Dict[str, int] = None
    window_maximized: bool = False
    
    # Recent files
    recent_files: List[str] = None
    
    # Analysis settings
    default_manual_threshold: float = 5.0
    default_noise_floor: float = 10.0
    frame_cache_size: int = 100
    
    # UI preferences
    theme: str = "dark"
    auto_save_results: bool = True
    show_progress_details: bool = True
    
    # Video settings
    default_jump_frames: int = 10
    auto_detect_on_load: bool = False
    
    # Export settings
    default_output_format: str = "csv"
    plot_dpi: int = 300
    plot_style: str = "default"
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.window_geometry is None:
            self.window_geometry = {}
        if self.recent_files is None:
            self.recent_files = []


class SettingsManager:
    """Manages application settings and configuration."""
    
    def __init__(self, settings_file: str = DEFAULT_SETTINGS_FILE):
        self.settings_file = settings_file
        self.settings = AppSettings()
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                
                # Update settings with loaded data
                for key, value in data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
                
                logging.info(f"Settings loaded from {self.settings_file}")
            else:
                logging.info("No settings file found, using defaults")
                
        except Exception as e:
            logging.warning(f"Could not load settings from {self.settings_file}: {e}")
            self.settings = AppSettings()  # Reset to defaults
    
    def save_settings(self):
        """Save current settings to file."""
        try:
            # Clean up recent files (remove non-existent files)
            self.settings.recent_files = [
                f for f in self.settings.recent_files 
                if os.path.exists(f)
            ]
            
            # Convert to dict and save
            settings_dict = asdict(self.settings)
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=2)
            
            logging.info(f"Settings saved to {self.settings_file}")
            
        except Exception as e:
            logging.warning(f"Could not save settings to {self.settings_file}: {e}")
    
    def get_recent_files(self) -> List[str]:
        """Get list of recent files."""
        return [f for f in self.settings.recent_files if os.path.exists(f)]
    
    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if not file_path or not os.path.exists(file_path):
            return
        
        # Remove if already exists
        if file_path in self.settings.recent_files:
            self.settings.recent_files.remove(file_path)
        
        # Add to beginning
        self.settings.recent_files.insert(0, file_path)
        
        # Limit list size
        self.settings.recent_files = self.settings.recent_files[:MAX_RECENT_FILES]
        
        # Auto-save
        self.save_settings()
    
    def remove_recent_file(self, file_path: str):
        """Remove file from recent files list."""
        if file_path in self.settings.recent_files:
            self.settings.recent_files.remove(file_path)
            self.save_settings()
    
    def clear_recent_files(self):
        """Clear all recent files."""
        self.settings.recent_files.clear()
        self.save_settings()
    
    def get_window_geometry(self) -> Dict[str, int]:
        """Get window geometry settings."""
        return self.settings.window_geometry.copy()
    
    def set_window_geometry(self, x: int, y: int, width: int, height: int):
        """Set window geometry settings."""
        self.settings.window_geometry = {
            'x': x, 'y': y, 'width': width, 'height': height
        }
        self.save_settings()
    
    def is_window_maximized(self) -> bool:
        """Check if window should be maximized."""
        return self.settings.window_maximized
    
    def set_window_maximized(self, maximized: bool):
        """Set window maximized state."""
        self.settings.window_maximized = maximized
        self.save_settings()
    
    def get_analysis_defaults(self) -> Dict[str, Any]:
        """Get default analysis parameters."""
        return {
            'manual_threshold': self.settings.default_manual_threshold,
            'noise_floor': self.settings.default_noise_floor,
            'jump_frames': self.settings.default_jump_frames,
            'auto_detect_on_load': self.settings.auto_detect_on_load
        }
    
    def set_analysis_defaults(self, **kwargs):
        """Set default analysis parameters."""
        if 'manual_threshold' in kwargs:
            self.settings.default_manual_threshold = kwargs['manual_threshold']
        if 'noise_floor' in kwargs:
            self.settings.default_noise_floor = kwargs['noise_floor']
        if 'jump_frames' in kwargs:
            self.settings.default_jump_frames = kwargs['jump_frames']
        if 'auto_detect_on_load' in kwargs:
            self.settings.auto_detect_on_load = kwargs['auto_detect_on_load']
        
        self.save_settings()
    
    def get_ui_preferences(self) -> Dict[str, Any]:
        """Get UI preferences."""
        return {
            'theme': self.settings.theme,
            'auto_save_results': self.settings.auto_save_results,
            'show_progress_details': self.settings.show_progress_details,
            'frame_cache_size': self.settings.frame_cache_size
        }
    
    def set_ui_preferences(self, **kwargs):
        """Set UI preferences."""
        if 'theme' in kwargs:
            self.settings.theme = kwargs['theme']
        if 'auto_save_results' in kwargs:
            self.settings.auto_save_results = kwargs['auto_save_results']
        if 'show_progress_details' in kwargs:
            self.settings.show_progress_details = kwargs['show_progress_details']
        if 'frame_cache_size' in kwargs:
            self.settings.frame_cache_size = kwargs['frame_cache_size']
        
        self.save_settings()
    
    def get_export_preferences(self) -> Dict[str, Any]:
        """Get export preferences."""
        return {
            'default_output_format': self.settings.default_output_format,
            'plot_dpi': self.settings.plot_dpi,
            'plot_style': self.settings.plot_style
        }
    
    def set_export_preferences(self, **kwargs):
        """Set export preferences."""
        if 'default_output_format' in kwargs:
            self.settings.default_output_format = kwargs['default_output_format']
        if 'plot_dpi' in kwargs:
            self.settings.plot_dpi = kwargs['plot_dpi']
        if 'plot_style' in kwargs:
            self.settings.plot_style = kwargs['plot_style']
        
        self.save_settings()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = AppSettings()
        self.save_settings()
        logging.info("Settings reset to defaults")
    
    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file."""
        try:
            settings_dict = asdict(self.settings)
            with open(file_path, 'w') as f:
                json.dump(settings_dict, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate and import settings
            imported_settings = AppSettings()
            for key, value in data.items():
                if hasattr(imported_settings, key):
                    setattr(imported_settings, key, value)
            
            self.settings = imported_settings
            self.save_settings()
            return True
            
        except Exception as e:
            logging.error(f"Error importing settings: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return getattr(self.settings, key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a specific setting value."""
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)
            self.save_settings()
        else:
            logging.warning(f"Unknown setting key: {key}")
    
    def validate_settings(self) -> List[str]:
        """Validate current settings and return list of issues."""
        issues = []
        
        # Validate numeric ranges
        if self.settings.default_manual_threshold < 0:
            issues.append("Manual threshold cannot be negative")
        
        if self.settings.default_noise_floor < 0:
            issues.append("Noise floor cannot be negative")
        
        if self.settings.frame_cache_size < 1:
            issues.append("Frame cache size must be at least 1")
        
        if self.settings.default_jump_frames < 1:
            issues.append("Jump frames must be at least 1")
        
        if self.settings.plot_dpi < 72:
            issues.append("Plot DPI must be at least 72")
        
        # Validate string values
        valid_themes = ["dark", "light"]
        if self.settings.theme not in valid_themes:
            issues.append(f"Invalid theme: {self.settings.theme}")
        
        valid_formats = ["csv", "json", "excel"]
        if self.settings.default_output_format not in valid_formats:
            issues.append(f"Invalid output format: {self.settings.default_output_format}")
        
        # Validate recent files
        valid_recent_files = []
        for file_path in self.settings.recent_files:
            if os.path.exists(file_path):
                valid_recent_files.append(file_path)
            else:
                issues.append(f"Recent file no longer exists: {file_path}")
        
        # Update recent files if any were invalid
        if len(valid_recent_files) != len(self.settings.recent_files):
            self.settings.recent_files = valid_recent_files
            self.save_settings()
        
        return issues
    
    def __del__(self):
        """Ensure settings are saved when object is destroyed."""
        try:
            self.save_settings()
        except:
            pass  # Ignore errors during cleanup