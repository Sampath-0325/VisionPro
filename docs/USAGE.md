# VisionPro Studio — Usage Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Single Image Processing](#single-image-processing)
- [Batch Processing](#batch-processing)
- [Selective Operations](#selective-operations)
- [Report Generation](#report-generation)
- [Visualization Options](#visualization-options)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process your first image
python app.py --input images/Arches.jpeg
```

This will:
1. Load the image
2. Apply all 40+ processing operations
3. Save results to `outputs/`
4. Generate reports in `reports/`
5. Create visualization dashboards

---

## Single Image Processing

### Basic Usage

```bash
python app.py --input path/to/your/image.jpg
```

### With Custom Output Directory

```bash
python app.py --input image.jpg --output my_results/
```

### With PNG Output Format

```bash
python app.py --input image.jpg --format png
```

### Verbose Mode

```bash
python app.py --input image.jpg --verbose
```

This enables DEBUG-level logging, showing detailed timing for each stage.

---

## Batch Processing

Process every supported image in a directory:

```bash
python app.py --batch images/
```

Each image gets its own subfolder in the output directory:

```
outputs/
├── Arches/
│   ├── grayscale/
│   ├── sobel/
│   └── ...
├── apple/
│   ├── grayscale/
│   └── ...
└── ...
```

### Error Handling

Batch processing continues even if individual images fail.
A summary is printed at the end showing successes and failures.

---

## Selective Operations

Run only specific processing stages:

```bash
# Just grayscale and edge detection
python app.py -i image.jpg --operations preprocessing,edge_detection

# Only metrics (no image output)
python app.py -i image.jpg --operations metrics --no-viz --report json
```

### Available Operations

| Operation | What It Does |
|:----------|:-------------|
| `preprocessing` | Grayscale, histogram eq., brightness, contrast, gamma |
| `filters` | Gaussian/median/bilateral blur, sharpen, emboss, sepia, vignette, HDR, cartoon |
| `edge_detection` | Sobel, Laplacian, Canny, Prewitt, Roberts, Scharr |
| `thresholding` | Binary, adaptive, Otsu thresholding |
| `morphology` | Erosion, dilation, opening, closing, gradient |
| `contours` | Contour detection, bounding boxes, convex hull |
| `color_analysis` | Dominant colors, palette, temperature, saturation |
| `metrics` | Brightness, contrast, sharpness, PSNR, SSIM |

---

## Report Generation

### Available Formats

```bash
# Text report
python app.py -i image.jpg --report txt

# Markdown report  
python app.py -i image.jpg --report md

# Styled HTML report (dark theme)
python app.py -i image.jpg --report html

# CSV metrics export
python app.py -i image.jpg --report csv

# JSON data export
python app.py -i image.jpg --report json

# All formats
python app.py -i image.jpg --report all

# No reports
python app.py -i image.jpg --report none
```

Reports are saved to the `reports/` directory.

---

## Visualization Options

By default, these visualizations are generated:

- **Processing Dashboard** — Grid of all processed images
- **RGB Histogram** — Color channel distribution
- **Grayscale Histogram** — Intensity distribution with mean line
- **Before/After** — Original vs. grayscale comparison
- **Metrics Overlay** — Image with key metrics displayed

### Skip Visualizations

```bash
python app.py -i image.jpg --no-viz
```

---

## Configuration

### JSON Config File

Create a `config.json` in the project root:

```json
{
    "gaussian_kernel": [7, 7],
    "median_kernel": 7,
    "canny_threshold1": 80,
    "canny_threshold2": 180,
    "threshold": 128,
    "morph_kernel": [3, 3],
    "figure_dpi": 150,
    "dominant_colors_k": 8,
    "log_level": "DEBUG"
}
```

Load it in your Python code:

```python
from src.config import Config
Config.load_from_json("config.json")
```

---

## Python API

### Using as a Library

```python
from src import (
    ProcessingPipeline,
    ImageFilters,
    EdgeDetector,
    ImageMetrics,
    Visualization,
)

# Load and process
pipeline = ProcessingPipeline()
pipeline.load_image("images/Arches.jpeg")

# Run all operations
pipeline.process_all()

# Or run selectively
pipeline.process_selected(["preprocessing", "edge_detection"])

# Get results
results = pipeline.get_results()
stats = pipeline.get_statistics()
timings = pipeline.get_timings()
```

### Using Individual Modules

```python
import cv2
from src.filters import ImageFilters
from src.edge_detection import EdgeDetector
from src.metrics import ImageMetrics

image = cv2.imread("image.jpg")

# Apply a single filter
blurred = ImageFilters.gaussian_blur(image)
cartoon = ImageFilters.cartoon(image)
sepia = ImageFilters.sepia(image)

# Edge detection
edges = EdgeDetector.canny(image, 100, 200)
overlay = EdgeDetector.edge_overlay(image)

# Compute metrics
psnr = ImageMetrics.psnr(image, blurred)
ssim = ImageMetrics.ssim(image, blurred)
colors = ImageMetrics.dominant_colors(image, k=5)
```

### Color Analysis

```python
from src.color_analysis import ColorAnalysis

colors, pcts = ColorAnalysis.dominant_colors(image, k=5)
palette_img = ColorAnalysis.create_palette(colors, pcts)
distribution = ColorAnalysis.color_distribution(image)
temperature = ColorAnalysis.color_temperature(image)
```

### Batch Processing

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor(output_dir="batch_output")
summary = processor.process_batch("images/")
processor.print_summary()
```

---

## Troubleshooting

### Common Issues

**Import Error: No module named 'cv2'**
```bash
pip install opencv-python
```

**Image not found**
- Check the file path is correct
- Ensure the format is supported (.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp, .gif)

**GIF loading fails**
```bash
pip install pillow
```

**Tests fail**
```bash
pip install pytest
python -m pytest tests/ -v
```

### Supported Image Formats

| Format | Extensions |
|:-------|:-----------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| BMP | `.bmp` |
| TIFF | `.tif`, `.tiff` |
| WebP | `.webp` |
| GIF | `.gif` (first frame) |
