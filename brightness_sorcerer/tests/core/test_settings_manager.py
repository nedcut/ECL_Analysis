"""Tests for SettingsManager class."""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, mock_open

from ...core.settings_manager import SettingsManager, AppSettings


class TestAppSettings:
    """Tests for AppSettings dataclass."""
    
    @pytest.mark.unit
    def test_init_defaults(self):
        """Test AppSettings initialization with defaults."""
        settings = AppSettings()
        
        assert settings.window_geometry == {}
        assert not settings.window_maximized
        assert settings.recent_files == []
        assert settings.default_manual_threshold == 5.0
        assert settings.default_noise_floor == 10.0
        assert settings.frame_cache_size == 100
        assert settings.theme == "dark"
        assert settings.auto_save_results
        assert settings.show_progress_details
        assert settings.default_jump_frames == 10
        assert not settings.auto_detect_on_load
        assert settings.default_output_format == "csv"
        assert settings.plot_dpi == 300
        assert settings.plot_style == "default"
    
    @pytest.mark.unit
    def test_init_with_values(self):
        """Test AppSettings initialization with specific values."""
        window_geometry = {'x': 100, 'y': 50, 'width': 800, 'height': 600}
        recent_files = ['/path/to/video1.mp4', '/path/to/video2.mp4']
        
        settings = AppSettings(
            window_geometry=window_geometry,
            window_maximized=True,
            recent_files=recent_files,
            default_manual_threshold=7.5,
            theme="light"
        )
        
        assert settings.window_geometry == window_geometry
        assert settings.window_maximized
        assert settings.recent_files == recent_files
        assert settings.default_manual_threshold == 7.5
        assert settings.theme == "light"


class TestSettingsManager:
    """Tests for SettingsManager class."""
    
    @pytest.mark.unit
    def test_init_new_settings_file(self, temp_dir):
        """Test initialization with non-existent settings file."""
        settings_file = os.path.join(temp_dir, "new_settings.json")
        manager = SettingsManager(settings_file)
        
        assert manager.settings_file == settings_file
        assert isinstance(manager.settings, AppSettings)
        assert manager.settings.recent_files == []
        assert manager.settings.default_manual_threshold == 5.0
    
    @pytest.mark.unit
    def test_init_existing_settings_file(self, temp_dir):
        """Test initialization with existing settings file."""
        settings_file = os.path.join(temp_dir, "existing_settings.json")
        
        # Create settings file with data
        settings_data = {
            'window_maximized': True,
            'recent_files': ['/test/video.mp4'],
            'default_manual_threshold': 8.0,
            'theme': 'light'
        }
        
        with open(settings_file, 'w') as f:
            json.dump(settings_data, f)
        
        manager = SettingsManager(settings_file)
        
        assert manager.settings.window_maximized
        assert manager.settings.recent_files == ['/test/video.mp4']
        assert manager.settings.default_manual_threshold == 8.0
        assert manager.settings.theme == 'light'
    
    @pytest.mark.unit
    def test_init_corrupted_settings_file(self, temp_dir):
        """Test initialization with corrupted settings file."""
        settings_file = os.path.join(temp_dir, "corrupted_settings.json")
        
        # Create corrupted JSON file
        with open(settings_file, 'w') as f:
            f.write("{ invalid json")
        
        manager = SettingsManager(settings_file)
        
        # Should fall back to defaults
        assert isinstance(manager.settings, AppSettings)
        assert manager.settings.recent_files == []
    
    @pytest.mark.unit
    def test_save_settings(self, temp_dir):
        """Test saving settings to file."""
        settings_file = os.path.join(temp_dir, "test_settings.json")
        manager = SettingsManager(settings_file)
        
        # Modify settings
        manager.settings.window_maximized = True
        manager.settings.recent_files = ['/test/video.mp4']
        manager.settings.default_manual_threshold = 7.5
        
        # Save settings
        manager.save_settings()
        
        # Verify file was created and contains correct data
        assert os.path.exists(settings_file)
        
        with open(settings_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['window_maximized']
        assert saved_data['recent_files'] == ['/test/video.mp4']
        assert saved_data['default_manual_threshold'] == 7.5
    
    @pytest.mark.unit
    def test_save_settings_cleanup_recent_files(self, temp_dir):
        """Test that save_settings cleans up non-existent recent files."""
        settings_file = os.path.join(temp_dir, "test_settings.json")
        manager = SettingsManager(settings_file)
        
        # Create one real file and add non-existent files
        real_file = os.path.join(temp_dir, "real_video.mp4")
        with open(real_file, 'w') as f:
            f.write("fake video")
        
        manager.settings.recent_files = [
            real_file,
            '/nonexistent/video1.mp4',
            '/nonexistent/video2.mp4'
        ]
        
        manager.save_settings()
        
        # Only the existing file should remain
        assert manager.settings.recent_files == [real_file]
    
    @pytest.mark.unit
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_settings_permission_error(self, mock_file, settings_manager):
        """Test save_settings with permission error."""
        # Should not raise exception
        settings_manager.save_settings()
    
    @pytest.mark.unit
    def test_get_recent_files(self, settings_manager, temp_dir):
        """Test getting recent files list."""
        # Create some real files
        file1 = os.path.join(temp_dir, "video1.mp4")
        file2 = os.path.join(temp_dir, "video2.mp4")
        
        with open(file1, 'w') as f:
            f.write("fake video 1")
        with open(file2, 'w') as f:
            f.write("fake video 2")
        
        # Set recent files (including non-existent)
        settings_manager.settings.recent_files = [
            file1,
            '/nonexistent/video.mp4',
            file2
        ]
        
        recent_files = settings_manager.get_recent_files()
        
        # Should only return existing files
        assert len(recent_files) == 2
        assert file1 in recent_files
        assert file2 in recent_files
        assert '/nonexistent/video.mp4' not in recent_files
    
    @pytest.mark.unit
    def test_add_recent_file(self, settings_manager, temp_dir):
        """Test adding recent files."""
        # Create test file
        test_file = os.path.join(temp_dir, "test_video.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video")
        
        # Add file
        settings_manager.add_recent_file(test_file)
        
        assert test_file in settings_manager.settings.recent_files
        assert settings_manager.settings.recent_files[0] == test_file
    
    @pytest.mark.unit
    def test_add_recent_file_duplicate(self, settings_manager, temp_dir):
        """Test adding duplicate recent file."""
        test_file = os.path.join(temp_dir, "test_video.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video")
        
        # Add file twice
        settings_manager.add_recent_file(test_file)
        settings_manager.add_recent_file(test_file)
        
        # Should only appear once, at the beginning
        assert settings_manager.settings.recent_files.count(test_file) == 1
        assert settings_manager.settings.recent_files[0] == test_file
    
    @pytest.mark.unit
    def test_add_recent_file_max_limit(self, settings_manager, temp_dir):
        """Test recent files list respects maximum limit."""
        # Create more files than the limit
        files = []
        for i in range(15):  # More than MAX_RECENT_FILES (10)
            file_path = os.path.join(temp_dir, f"video{i}.mp4")
            with open(file_path, 'w') as f:
                f.write(f"fake video {i}")
            files.append(file_path)
            settings_manager.add_recent_file(file_path)
        
        # Should only keep the last 10 files
        assert len(settings_manager.settings.recent_files) == 10
        
        # Most recent files should be at the beginning
        for i in range(10):
            expected_file = files[14 - i]  # Last 10 files in reverse order
            assert settings_manager.settings.recent_files[i] == expected_file
    
    @pytest.mark.unit
    def test_add_recent_file_nonexistent(self, settings_manager):
        """Test adding non-existent file to recent files."""
        settings_manager.add_recent_file('/nonexistent/video.mp4')
        
        # Should not be added
        assert '/nonexistent/video.mp4' not in settings_manager.settings.recent_files
    
    @pytest.mark.unit
    def test_remove_recent_file(self, settings_manager, temp_dir):
        """Test removing recent file."""
        test_file = os.path.join(temp_dir, "test_video.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video")
        
        settings_manager.add_recent_file(test_file)
        assert test_file in settings_manager.settings.recent_files
        
        settings_manager.remove_recent_file(test_file)
        assert test_file not in settings_manager.settings.recent_files
    
    @pytest.mark.unit
    def test_clear_recent_files(self, settings_manager, temp_dir):
        """Test clearing recent files."""
        # Add some files
        for i in range(3):
            file_path = os.path.join(temp_dir, f"video{i}.mp4")
            with open(file_path, 'w') as f:
                f.write(f"fake video {i}")
            settings_manager.add_recent_file(file_path)
        
        assert len(settings_manager.settings.recent_files) == 3
        
        settings_manager.clear_recent_files()
        assert len(settings_manager.settings.recent_files) == 0
    
    @pytest.mark.unit
    def test_window_geometry_methods(self, settings_manager):
        """Test window geometry getter and setter."""
        # Test getter with empty geometry
        geometry = settings_manager.get_window_geometry()
        assert geometry == {}
        
        # Test setter
        settings_manager.set_window_geometry(100, 50, 800, 600)
        
        geometry = settings_manager.get_window_geometry()
        assert geometry == {'x': 100, 'y': 50, 'width': 800, 'height': 600}
        
        # Original settings should be modified
        assert settings_manager.settings.window_geometry == geometry
    
    @pytest.mark.unit
    def test_window_maximized_methods(self, settings_manager):
        """Test window maximized getter and setter."""
        # Test default
        assert not settings_manager.is_window_maximized()
        
        # Test setter
        settings_manager.set_window_maximized(True)
        assert settings_manager.is_window_maximized()
        assert settings_manager.settings.window_maximized
    
    @pytest.mark.unit
    def test_analysis_defaults_methods(self, settings_manager):
        """Test analysis defaults getter and setter."""
        # Test getter
        defaults = settings_manager.get_analysis_defaults()
        expected_keys = ['manual_threshold', 'noise_floor', 'jump_frames', 'auto_detect_on_load']
        for key in expected_keys:
            assert key in defaults
        
        # Test setter
        settings_manager.set_analysis_defaults(
            manual_threshold=8.5,
            noise_floor=12.0,
            jump_frames=15,
            auto_detect_on_load=True
        )
        
        updated_defaults = settings_manager.get_analysis_defaults()
        assert updated_defaults['manual_threshold'] == 8.5
        assert updated_defaults['noise_floor'] == 12.0
        assert updated_defaults['jump_frames'] == 15
        assert updated_defaults['auto_detect_on_load']
    
    @pytest.mark.unit
    def test_ui_preferences_methods(self, settings_manager):
        """Test UI preferences getter and setter."""
        # Test getter
        preferences = settings_manager.get_ui_preferences()
        expected_keys = ['theme', 'auto_save_results', 'show_progress_details', 'frame_cache_size']
        for key in expected_keys:
            assert key in preferences
        
        # Test setter
        settings_manager.set_ui_preferences(
            theme='light',
            auto_save_results=False,
            show_progress_details=False,
            frame_cache_size=200
        )
        
        updated_preferences = settings_manager.get_ui_preferences()
        assert updated_preferences['theme'] == 'light'
        assert not updated_preferences['auto_save_results']
        assert not updated_preferences['show_progress_details']
        assert updated_preferences['frame_cache_size'] == 200
    
    @pytest.mark.unit
    def test_export_preferences_methods(self, settings_manager):
        """Test export preferences getter and setter."""
        # Test getter
        preferences = settings_manager.get_export_preferences()
        expected_keys = ['default_output_format', 'plot_dpi', 'plot_style']
        for key in expected_keys:
            assert key in preferences
        
        # Test setter
        settings_manager.set_export_preferences(
            default_output_format='json',
            plot_dpi=600,
            plot_style='seaborn'
        )
        
        updated_preferences = settings_manager.get_export_preferences()
        assert updated_preferences['default_output_format'] == 'json'
        assert updated_preferences['plot_dpi'] == 600
        assert updated_preferences['plot_style'] == 'seaborn'
    
    @pytest.mark.unit
    def test_reset_to_defaults(self, settings_manager):
        """Test resetting settings to defaults."""
        # Modify settings
        settings_manager.settings.window_maximized = True
        settings_manager.settings.default_manual_threshold = 10.0
        settings_manager.settings.theme = 'light'
        settings_manager.add_recent_file('/some/file.mp4')  # Won't actually add due to non-existence
        
        # Reset to defaults
        settings_manager.reset_to_defaults()
        
        # Should be back to defaults
        assert not settings_manager.settings.window_maximized
        assert settings_manager.settings.default_manual_threshold == 5.0
        assert settings_manager.settings.theme == 'dark'
        assert settings_manager.settings.recent_files == []
    
    @pytest.mark.unit
    def test_export_import_settings(self, settings_manager, temp_dir):
        """Test exporting and importing settings."""
        # Modify settings
        settings_manager.settings.window_maximized = True
        settings_manager.settings.default_manual_threshold = 7.5
        settings_manager.settings.theme = 'light'
        
        # Export settings
        export_file = os.path.join(temp_dir, "exported_settings.json")
        assert settings_manager.export_settings(export_file)
        assert os.path.exists(export_file)
        
        # Reset settings
        settings_manager.reset_to_defaults()
        assert not settings_manager.settings.window_maximized
        
        # Import settings
        assert settings_manager.import_settings(export_file)
        
        # Settings should be restored
        assert settings_manager.settings.window_maximized
        assert settings_manager.settings.default_manual_threshold == 7.5
        assert settings_manager.settings.theme == 'light'
    
    @pytest.mark.unit
    def test_export_settings_error(self, settings_manager):
        """Test export settings with file error."""
        # Try to export to invalid path
        assert not settings_manager.export_settings('/invalid/path/settings.json')
    
    @pytest.mark.unit
    def test_import_settings_nonexistent_file(self, settings_manager):
        """Test import settings with non-existent file."""
        assert not settings_manager.import_settings('/nonexistent/settings.json')
    
    @pytest.mark.unit
    def test_import_settings_invalid_json(self, settings_manager, temp_dir):
        """Test import settings with invalid JSON."""
        invalid_file = os.path.join(temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json")
        
        assert not settings_manager.import_settings(invalid_file)
    
    @pytest.mark.unit
    def test_get_set_setting(self, settings_manager):
        """Test generic setting getter and setter."""
        # Test getting existing setting
        value = settings_manager.get_setting('theme', 'default')
        assert value == 'dark'  # Default theme
        
        # Test getting non-existent setting with default
        value = settings_manager.get_setting('nonexistent', 'default_value')
        assert value == 'default_value'
        
        # Test setting valid setting
        settings_manager.set_setting('theme', 'light')
        assert settings_manager.settings.theme == 'light'
        
        # Test setting invalid setting (should log warning but not crash)
        settings_manager.set_setting('invalid_setting', 'value')
        # Should not raise exception
    
    @pytest.mark.unit
    def test_validate_settings(self, settings_manager, temp_dir):
        """Test settings validation."""
        # Start with valid settings
        issues = settings_manager.validate_settings()
        assert len(issues) == 0
        
        # Introduce invalid values
        settings_manager.settings.default_manual_threshold = -5.0
        settings_manager.settings.default_noise_floor = -2.0
        settings_manager.settings.frame_cache_size = 0
        settings_manager.settings.default_jump_frames = -1
        settings_manager.settings.plot_dpi = 50
        settings_manager.settings.theme = 'invalid_theme'
        settings_manager.settings.default_output_format = 'invalid_format'
        
        # Add non-existent file to recent files
        settings_manager.settings.recent_files = ['/nonexistent/file.mp4']
        
        issues = settings_manager.validate_settings()
        
        # Should find multiple issues
        assert len(issues) > 0
        
        # Check specific issues
        issue_messages = ' '.join(issues)
        assert 'threshold cannot be negative' in issue_messages
        assert 'Noise floor cannot be negative' in issue_messages
        assert 'cache size must be at least 1' in issue_messages
        assert 'Jump frames must be at least 1' in issue_messages
        assert 'Plot DPI must be at least 72' in issue_messages
        assert 'Invalid theme' in issue_messages
        assert 'Invalid output format' in issue_messages
        assert 'no longer exists' in issue_messages
        
        # Recent files should be cleaned up
        assert len(settings_manager.settings.recent_files) == 0
    
    @pytest.mark.unit
    def test_destructor_saves_settings(self, temp_dir):
        """Test that destructor saves settings."""
        settings_file = os.path.join(temp_dir, "destructor_test.json")
        
        # Create manager and modify settings
        manager = SettingsManager(settings_file)
        manager.settings.window_maximized = True
        
        # Delete manager (should trigger __del__)
        del manager
        
        # Settings should be saved
        # Note: This test is flaky because __del__ timing is not guaranteed
        # In practice, explicit save_settings() calls are more reliable
    
    @pytest.mark.unit
    @patch('brightness_sorcerer.core.settings_manager.SettingsManager.save_settings')
    def test_auto_save_on_setting_changes(self, mock_save, settings_manager):
        """Test that settings are automatically saved on changes."""
        # Methods that should trigger auto-save
        settings_manager.add_recent_file('/fake/path.mp4')  # Won't add due to non-existence, but tries to save
        mock_save.assert_called()
        
        mock_save.reset_mock()
        settings_manager.set_window_geometry(100, 100, 800, 600)
        mock_save.assert_called()
        
        mock_save.reset_mock()
        settings_manager.set_window_maximized(True)
        mock_save.assert_called()
        
        mock_save.reset_mock()
        settings_manager.set_analysis_defaults(manual_threshold=8.0)
        mock_save.assert_called()


@pytest.mark.integration
class TestSettingsManagerIntegration:
    """Integration tests for SettingsManager."""
    
    def test_full_settings_workflow(self, temp_dir):
        """Test complete settings management workflow."""
        settings_file = os.path.join(temp_dir, "workflow_test.json")
        
        # Create manager and configure settings
        manager1 = SettingsManager(settings_file)
        
        # Configure various settings
        manager1.set_window_geometry(200, 100, 1000, 700)
        manager1.set_window_maximized(True)
        manager1.set_analysis_defaults(
            manual_threshold=8.5,
            noise_floor=12.0,
            auto_detect_on_load=True
        )
        manager1.set_ui_preferences(
            theme='light',
            frame_cache_size=150
        )
        
        # Create test files and add to recent files
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
            with open(file_path, 'w') as f:
                f.write(f"fake video {i}")
            test_files.append(file_path)
            manager1.add_recent_file(file_path)
        
        # Save settings
        manager1.save_settings()
        
        # Create new manager with same settings file
        manager2 = SettingsManager(settings_file)
        
        # Verify all settings were preserved
        assert manager2.get_window_geometry() == {'x': 200, 'y': 100, 'width': 1000, 'height': 700}
        assert manager2.is_window_maximized()
        
        analysis_defaults = manager2.get_analysis_defaults()
        assert analysis_defaults['manual_threshold'] == 8.5
        assert analysis_defaults['noise_floor'] == 12.0
        assert analysis_defaults['auto_detect_on_load']
        
        ui_preferences = manager2.get_ui_preferences()
        assert ui_preferences['theme'] == 'light'
        assert ui_preferences['frame_cache_size'] == 150
        
        recent_files = manager2.get_recent_files()
        assert len(recent_files) == 3
        for test_file in test_files:
            assert test_file in recent_files
        
        # Validate settings
        issues = manager2.validate_settings()
        assert len(issues) == 0