#!/usr/bin/env python3
"""
Brightness Sorcerer Launcher
Professional Video Brightness Analysis Tool

This is a convenience launcher that provides additional command-line options
and better error handling for the main application.
"""

import argparse
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup the environment for running Brightness Sorcerer."""
    # Add the current directory to the Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5>=5.15.0")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python>=4.5.0")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas>=1.3.0")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy>=1.21.0")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib>=3.4.0")
    
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies with:")
        print("  pip install " + " ".join(missing_deps))
        print("\nOr install all requirements with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        prog="brightness-sorcerer",
        description="Brightness Sorcerer v2.0 - Professional Video Brightness Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  brightness-sorcerer                    # Launch GUI
  brightness-sorcerer --version          # Show version
  brightness-sorcerer --check-deps       # Check dependencies
  brightness-sorcerer --log-level DEBUG  # Enable debug logging
  
For more information, visit: https://github.com/brightness-sorcerer/brightness-sorcerer
        """
    )
    
    parser.add_argument(
        "--version", 
        action="store_true", 
        help="Show version information and exit"
    )
    
    parser.add_argument(
        "--check-deps", 
        action="store_true", 
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        help="Write logs to specified file"
    )
    
    parser.add_argument(
        "--config-dir", 
        type=str, 
        help="Use custom configuration directory"
    )
    
    parser.add_argument(
        "--no-audio", 
        action="store_true", 
        help="Disable audio features even if libraries are available"
    )
    
    parser.add_argument(
        "video_file", 
        nargs="?", 
        help="Video file to open on startup"
    )
    
    args = parser.parse_args()
    
    # Handle version request
    if args.version:
        setup_environment()
        try:
            from main import __version__, __author__, __description__
            print(f"Brightness Sorcerer v{__version__}")
            print(f"Author: {__author__}")
            print(f"Description: {__description__}")
            print("\nCore Dependencies:")
            import PyQt5
            import cv2
            import pandas
            import numpy
            import matplotlib
            print(f"  PyQt5: {PyQt5.QtCore.PYQT_VERSION_STR}")
            print(f"  OpenCV: {cv2.__version__}")
            print(f"  Pandas: {pandas.__version__}")
            print(f"  NumPy: {numpy.__version__}")
            print(f"  Matplotlib: {matplotlib.__version__}")
        except ImportError as e:
            print(f"Error importing modules: {e}")
            return 1
        return 0
    
    # Handle dependency check
    if args.check_deps:
        print("Checking dependencies...")
        if check_dependencies():
            print("✓ All required dependencies are available")
            
            # Check optional dependencies
            optional_deps = {}
            try:
                import pygame
                optional_deps["pygame"] = pygame.version.ver
            except ImportError:
                optional_deps["pygame"] = "NOT AVAILABLE"
            
            try:
                import librosa
                optional_deps["librosa"] = librosa.__version__
            except ImportError:
                optional_deps["librosa"] = "NOT AVAILABLE"
            
            try:
                import psutil
                optional_deps["psutil"] = psutil.__version__
            except ImportError:
                optional_deps["psutil"] = "NOT AVAILABLE"
            
            print("\nOptional Dependencies:")
            for dep, version in optional_deps.items():
                status = "✓" if version != "NOT AVAILABLE" else "✗"
                print(f"  {status} {dep}: {version}")
                
            return 0
        else:
            return 1
    
    # Setup environment
    setup_environment()
    
    # Check dependencies before launching
    if not check_dependencies():
        return 1
    
    # Set environment variables based on arguments
    if args.config_dir:
        os.environ["BS_CONFIG_DIR"] = args.config_dir
    
    if args.no_audio:
        os.environ["BS_DISABLE_AUDIO"] = "1"
    
    # Import and run the main application
    try:
        from main import main as app_main
        
        # Override logging setup if specified
        if args.log_level or args.log_file:
            from main import setup_logging
            setup_logging(args.log_level, args.log_file)
        
        # Run the application
        return app_main()
        
    except ImportError as e:
        print(f"Error importing main application: {e}")
        print("Make sure main.py is in the same directory as this launcher.")
        return 1
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())