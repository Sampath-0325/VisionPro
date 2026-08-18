# VisionPro Studio

<div align="center">

**Advanced Image Processing & Analysis Toolkit**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Version](https://img.shields.io/badge/Version-2.0.0-purple)

*A professional-grade image processing pipeline with 40+ operations,
batch processing, multi-format reports, and beautiful visualizations.*

</div>

---

##  Features

| Category | Operations |
|:---------|:-----------|
| **Preprocessing** | Grayscale, Histogram Equalization, CLAHE, Brightness, Contrast, Gamma, Resize, Crop, Rotate, Flip |
| **Filters** | Gaussian, Median, Bilateral, Box Blur, Sharpen, Unsharp Mask, Emboss, Sepia, Negative, Sketch, Vignette, HDR, Cartoon, Oil Painting, Color Quantize, Warm/Cool Tone |
| **Edge Detection** | Sobel, Laplacian, Canny, Prewitt, Roberts, Scharr, Edge Overlay, Multi-scale Canny |
| **Thresholding** | Binary, Binary Inverse, Truncate, To-Zero, Adaptive Mean, Adaptive Gaussian, Otsu, Otsu+Gaussian |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, Top Hat, Black Hat, Hit-or-Miss |
| **Contour Detection** | Find/Draw Contours, Bounding Boxes, Rotated Boxes, Convex Hull, Polygon Approximation, Enclosing Circles |
| **Color Analysis** | Dominant Colors (K-Means), Color Palette, Color Temperature, Saturation Analysis, Channel Distribution, Histogram Comparison |
| **Metrics** | Brightness, Contrast, Sharpness, Entropy, PSNR, SSIM, Mean/Median/Variance, Dominant Colors |
| **Reports** | TXT, Markdown, HTML (dark-themed), CSV, JSON |
| **Visualization** | Dark-themed Dashboards, RGB/Grayscale Histograms, Before/After Comparison, Color Palette, Metrics Overlay |

---

##  Project Structure

```
VisionPro-Studio/
├── app.py                    # CLI entry point
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── LICENSE                   # MIT License
│
├── src/                      # Core modules
│   ├── __init__.py
│   ├── config.py             # Centralized configuration
│   ├── image_loader.py       # Image loading, GIF, EXIF
│   ├── preprocessing.py      # Grayscale, resize, rotate, noise
│   ├── filters.py            # Blur, sharpen, artistic effects
│   ├── edge_detection.py     # Sobel, Canny, Scharr, etc.
│   ├── thresholding.py       # Binary, adaptive, Otsu
│   ├── morphology.py         # Erosion, dilation, etc.
│   ├── contour_detection.py  # Contour analysis
│   ├── color_analysis.py     # Dominant colors, palettes
│   ├── metrics.py            # PSNR, SSIM, statistics
│   ├── visualization.py      # Charts, dashboards
│   ├── report_generator.py   # TXT/MD/HTML/CSV/JSON reports
│   ├── batch_processor.py    # Multi-image processing
│   ├── logger.py             # Colored logging system
│   └── utils.py              # Utilities & helpers
│
├── tests/                    # Test suite (pytest)
│   ├── conftest.py
│   ├── test_image_loader.py
│   ├── test_preprocessing.py
│   ├── test_filters.py
│   ├── test_edge_detection.py
│   ├── test_thresholding.py
│   ├── test_morphology.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   └── test_utils.py
│
├── images/                   # Input images
├── outputs/                  # Generated results
├── reports/                  # Generated reports
├── docs/                     # Documentation
│   ├── USAGE.md
│   └── API.md
├── assets/                   # Themes, icons
└── logs/                     # Processing logs
```

---

##  Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/visionpro-studio/VisionPro-Studio.git
cd VisionPro-Studio

# Install dependencies
pip install -r requirements.txt
```

### Development Install

```bash
pip install -e ".[dev]"
```

---

##  Usage

### Single Image Processing

```bash
# Process with all operations
python app.py --input images/Arches.jpeg

# Short form
python app.py -i images/Arches.jpeg
```

### Selective Operations

```bash
# Only run specific operations
python app.py -i images/Arches.jpeg --operations preprocessing,edge_detection,metrics
```

### Batch Processing

```bash
# Process all images in a directory
python app.py --batch images/

# With custom output directory
python app.py --batch images/ --output results/
```

### Report Formats

```bash
# Generate only HTML report
python app.py -i images/Arches.jpeg --report html

# Generate all report formats
python app.py -i images/Arches.jpeg --report all

# Skip reports
python app.py -i images/Arches.jpeg --report none
```

### Other Options

```bash
# Skip visualization generation
python app.py -i images/Arches.jpeg --no-viz

# Verbose/debug output
python app.py -i images/Arches.jpeg --verbose

# PNG output format
python app.py -i images/Arches.jpeg --format png

# List all available operations
python app.py --list-operations

# Show version
python app.py --version
```

---

##  Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_filters.py -v

# Run with coverage
python -m pytest tests/ --tb=short
```

---

##  Configuration

### CLI Arguments

| Argument | Short | Description |
|:---------|:------|:------------|
| `--input` | `-i` | Single image path |
| `--batch` | `-b` | Directory for batch processing |
| `--output` | `-o` | Custom output directory |
| `--operations` | | Comma-separated operation list |
| `--format` | | Output format: `jpg` or `png` |
| `--report` | | Report format: `txt`, `md`, `html`, `csv`, `json`, `all`, `none` |
| `--no-viz` | | Skip visualization generation |
| `--verbose` | `-v` | Enable debug logging |
| `--list-operations` | | Show available operations |
| `--version` | | Show version |

### Available Operations

- `preprocessing` — Grayscale, histogram equalization, brightness, contrast, gamma
- `filters` — Blur, sharpen, artistic effects, vignette, HDR, cartoon
- `edge_detection` — Sobel, Laplacian, Canny, Prewitt, Roberts, Scharr
- `thresholding` — Binary, adaptive, Otsu thresholding
- `morphology` — Erosion, dilation, opening, closing, gradient
- `contours` — Contour detection and analysis
- `color_analysis` — Dominant colors, palette, temperature
- `metrics` — Image statistics and quality metrics

### JSON Configuration

You can also load settings from a JSON file:

```json
{
    "gaussian_kernel": [7, 7],
    "canny_threshold1": 80,
    "canny_threshold2": 180,
    "figure_dpi": 150,
    "log_level": "DEBUG"
}
```

---

##  Output Examples

After processing, results are organized in `outputs/`:

```
outputs/
├── grayscale/          # Grayscale conversions
├── histogram/          # Histogram equalization results
├── gaussian/           # Gaussian blur
├── median/             # Median filtering
├── bilateral/          # Bilateral filter
├── sharpen/            # Sharpening & effects
├── sobel/              # Sobel edge detection
├── laplacian/          # Laplacian edges
├── canny/              # Canny edge detection
├── threshold/          # Thresholding results
├── morphology/         # Morphological operations
├── contour/            # Contour detection
├── color/              # Color analysis
└── comparison/         # Dashboards & charts
    ├── dashboard.png
    ├── rgb_histogram.png
    ├── gray_histogram.png
    ├── before_after.png
    ├── metrics_overlay.png
    └── color_palette.png
```

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##  Author

**VisionPro Studio Team**

---

<div align="center">
<sub>Built with using Python, OpenCV, NumPy, and Matplotlib</sub>
</div>
