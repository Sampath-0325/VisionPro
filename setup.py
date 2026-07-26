"""
==============================================================
VisionPro Studio
Package Setup
==============================================================
"""

from setuptools import setup, find_packages
from pathlib import Path


# Read README for long description
readme_path = Path(__file__).parent / "README.md"

long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


setup(

    name="visionpro-studio",

    version="2.0.0",

    author="VisionPro Studio Team",

    description="Advanced Image Processing & Analysis Toolkit",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/visionpro-studio/VisionPro-Studio",

    packages=find_packages(),

    python_requires=">=3.9",

    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "visionpro=app:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],

    keywords="image-processing computer-vision opencv filters edge-detection",
)
