"""File handling utilities for Brightness Sorcerer."""

import os
from typing import List, Optional, Tuple
import logging

# Supported video formats
SUPPORTED_VIDEO_FORMATS = [
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.m4v', '.flv',
    '.webm', '.ogv', '.3gp', '.mpg', '.mpeg', '.ts', '.mts'
]

SUPPORTED_IMAGE_FORMATS = [
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'
]


def validate_video_file(file_path: str) -> bool:
    """Validate if file is a supported video format."""
    if not file_path or not os.path.exists(file_path):
        return False
    
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_VIDEO_FORMATS


def get_supported_formats() -> List[str]:
    """Get list of supported video formats."""
    return SUPPORTED_VIDEO_FORMATS.copy()


def get_video_filter_string() -> str:
    """Get file dialog filter string for video files."""
    video_exts = [f"*{ext}" for ext in SUPPORTED_VIDEO_FORMATS]
    return f"Video files ({' '.join(video_exts)})"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    if not filename:
        return "untitled"
    
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "untitled"
    
    return filename


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if necessary."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logging.info(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {e}")
        return False


def get_unique_filename(base_path: str, extension: str = "") -> str:
    """Get a unique filename by appending numbers if file exists."""
    if not extension.startswith('.') and extension:
        extension = '.' + extension
    
    full_path = base_path + extension
    
    if not os.path.exists(full_path):
        return full_path
    
    # Find unique name by appending numbers
    counter = 1
    while True:
        name_part = f"{base_path}_{counter:03d}"
        full_path = name_part + extension
        
        if not os.path.exists(full_path):
            return full_path
        
        counter += 1
        
        # Safety check to avoid infinite loop
        if counter > 9999:
            break
    
    # Fallback with timestamp
    import time
    timestamp = int(time.time())
    return f"{base_path}_{timestamp}{extension}"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
    except Exception as e:
        logging.error(f"Error getting file size for {file_path}: {e}")
    return 0.0


def get_available_disk_space_mb(directory: str) -> float:
    """Get available disk space in megabytes."""
    try:
        if os.path.exists(directory):
            stat = os.statvfs(directory) if hasattr(os, 'statvfs') else None
            if stat:
                # Unix/Linux/macOS
                available_bytes = stat.f_bavail * stat.f_frsize
                return available_bytes / (1024 * 1024)
            else:
                # Windows fallback
                import shutil
                _, _, free_bytes = shutil.disk_usage(directory)
                return free_bytes / (1024 * 1024)
    except Exception as e:
        logging.error(f"Error getting disk space for {directory}: {e}")
    return 0.0


def validate_output_directory(directory: str) -> Tuple[bool, str]:
    """Validate output directory and return status with message."""
    if not directory:
        return False, "No directory specified"
    
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create directory: {e}"
        
        # Check if directory is writable
        test_file = os.path.join(directory, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            return False, f"Directory is not writable: {e}"
        
        # Check available space (warn if less than 100MB)
        available_mb = get_available_disk_space_mb(directory)
        if available_mb < 100:
            return False, f"Low disk space: {available_mb:.1f} MB available"
        
        return True, f"Directory OK ({available_mb:.1f} MB available)"
        
    except Exception as e:
        return False, f"Directory validation error: {e}"


def backup_file(file_path: str, backup_suffix: str = ".bak") -> Optional[str]:
    """Create a backup copy of a file."""
    if not os.path.exists(file_path):
        return None
    
    try:
        backup_path = file_path + backup_suffix
        backup_path = get_unique_filename(backup_path.replace(backup_suffix, ''), backup_suffix)
        
        import shutil
        shutil.copy2(file_path, backup_path)
        logging.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logging.error(f"Error creating backup for {file_path}: {e}")
        return None


def cleanup_temp_files(directory: str, pattern: str = "temp_*") -> int:
    """Clean up temporary files matching pattern."""
    import glob
    
    try:
        temp_pattern = os.path.join(directory, pattern)
        temp_files = glob.glob(temp_pattern)
        
        cleaned_count = 0
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                cleaned_count += 1
            except Exception as e:
                logging.warning(f"Could not remove temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            logging.info(f"Cleaned up {cleaned_count} temporary files")
        
        return cleaned_count
        
    except Exception as e:
        logging.error(f"Error during temp file cleanup: {e}")
        return 0