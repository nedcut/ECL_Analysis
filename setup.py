#!/usr/bin/env python3
"""
Setup script for Brightness Sorcerer v2.0
Professional Video Brightness Analysis Tool
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we're using Python 3.7+
if sys.version_info < (3, 7):
    sys.exit("Brightness Sorcerer requires Python 3.7 or higher")

# Read version from main.py
def get_version():
    """Extract version from main.py"""
    version_file = os.path.join(os.path.dirname(__file__), 'main.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "2.0.0"

# Read README for long description
def get_long_description():
    """Read README file for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Professional Video Brightness Analysis Tool"

# Core requirements (always needed)
CORE_REQUIREMENTS = [
    'PyQt5>=5.15.0',
    'opencv-python>=4.5.0',
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'matplotlib>=3.4.0',
]

# Optional requirements for enhanced features
AUDIO_REQUIREMENTS = [
    'pygame>=2.0.0',
    'librosa>=0.10.0',
    'soundfile>=0.12.0',
]

PERFORMANCE_REQUIREMENTS = [
    'psutil>=5.8.0',  # For memory monitoring
]

DEVELOPMENT_REQUIREMENTS = [
    'pytest>=6.0.0',
    'pytest-qt>=4.0.0',
    'black>=21.0.0',
    'flake8>=3.9.0',
    'mypy>=0.910',
]

# All optional requirements combined
ALL_REQUIREMENTS = AUDIO_REQUIREMENTS + PERFORMANCE_REQUIREMENTS

setup(
    name="brightness-sorcerer",
    version=get_version(),
    author="Brightness Sorcerer Development Team",
    author_email="dev@brightnesssorcerer.dev",
    description="Professional Video Brightness Analysis Tool",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/brightness-sorcerer/brightness-sorcerer",
    project_urls={
        "Bug Reports": "https://github.com/brightness-sorcerer/brightness-sorcerer/issues",
        "Source": "https://github.com/brightness-sorcerer/brightness-sorcerer",
        "Documentation": "https://brightness-sorcerer.readthedocs.io/",
    },
    
    # Package configuration
    py_modules=["main"],
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        'audio': AUDIO_REQUIREMENTS,
        'performance': PERFORMANCE_REQUIREMENTS,
        'dev': DEVELOPMENT_REQUIREMENTS,
        'all': ALL_REQUIREMENTS + DEVELOPMENT_REQUIREMENTS,
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'brightness-sorcerer=main:main',
            'bs=main:main',  # Short alias
        ],
        'gui_scripts': [
            'brightness-sorcerer-gui=main:main',
        ],
    },
    
    # Package data
    package_data={
        '': ['*.md', '*.txt', '*.json', '*.ico', '*.png'],
    },
    include_package_data=True,
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization", 
        "Topic :: Multimedia :: Video :: Display",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
        "Environment :: Win32 (MS Windows)",
        "Environment :: MacOS X",
    ],
    
    # Keywords for discoverability
    keywords=[
        "video", "brightness", "analysis", "computer-vision", "opencv", 
        "pyqt5", "gui", "roi", "color-space", "lab", "visualization",
        "research", "measurement", "image-processing"
    ],
    
    # Metadata
    license="MIT",
    zip_safe=False,
    
    # Platform requirements
    platforms=["any"],
    
    # Additional metadata
    download_url="https://github.com/brightness-sorcerer/brightness-sorcerer/archive/v{}.tar.gz".format(get_version()),
)