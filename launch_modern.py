#!/usr/bin/env python3
"""
Modern Brightness Sorcerer Launcher

Launch the enhanced UI version of Brightness Sorcerer with improved architecture
and modern user experience.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from brightness_sorcerer.ui.main_window import create_application, ModernMainWindow
    
    def main():
        """Launch the modern Brightness Sorcerer application."""
        print("🌟 Starting Brightness Sorcerer v2.0 - Modern UI")
        print("Features:")
        print("  ✓ Enhanced video player with modern controls")
        print("  ✓ Modular UI architecture")
        print("  ✓ Professional dark theme")
        print("  ✓ Improved analysis panel")
        print("  ✓ Modern results visualization")
        print("  ✓ Responsive layout design")
        print("")
        
        # Create Qt application
        app = create_application()
        
        # Create and show main window
        window = ModernMainWindow()
        window.show()
        
        print("✅ Application started successfully!")
        print("💡 Drop a video file onto the video player to get started")
        print("")
        
        # Start event loop
        return app.exec_()
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install PyQt5 opencv-python numpy matplotlib pandas")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    sys.exit(1)