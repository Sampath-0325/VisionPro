# VisionPro Studio — API Reference

## Table of Contents

- [Config](#config)
- [ImageLoader](#imageloader)
- [Preprocessor](#preprocessor)
- [ImageFilters](#imagefilters)
- [EdgeDetector](#edgedetector)
- [Thresholding](#thresholding)
- [Morphology](#morphology)
- [ContourDetector](#contourdetector)
- [ColorAnalysis](#coloranalysis)
- [ImageMetrics](#imagemetrics)
- [Visualization](#visualization)
- [ReportGenerator](#reportgenerator)
- [ProcessingPipeline](#processingpipeline)
- [BatchProcessor](#batchprocessor)
- [Logger](#logger)
- [Utils](#utils)

---

## Config

**Module:** `src.config`

Centralized configuration class with all project settings.

### Class Methods

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `create_project_structure()` | — | `None` | Create all output directories |
| `from_args(args)` | `argparse.Namespace` | `Config` | Override config from CLI args |
| `validate()` | — | `list[str]` | Validate current configuration |
| `load_from_json(path)` | `str` | `None` | Load overrides from JSON file |
| `to_dict()` | — | `dict` | Export config as dictionary |

---

## ImageLoader

**Module:** `src.image_loader`

### Methods

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `load(filepath)` | `str \| Path` | `ndarray` | Load image (supports GIF) |
| `get_metadata()` | — | `dict` | Get extracted metadata |
| `display_metadata()` | — | `None` | Print metadata to console |

---

## Preprocessor

**Module:** `src.preprocessing`

All methods are `@staticmethod`.

### Color Conversions

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `grayscale(image)` | BGR image | Grayscale image |
| `to_hsv(image)` | BGR image | HSV image |
| `to_lab(image)` | BGR image | LAB image |
| `to_rgb(image)` | BGR image | RGB image |

### Enhancement

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `histogram_equalization(gray)` | Grayscale image | Enhanced image |
| `clahe(gray, clip_limit, tile_grid_size)` | Grayscale, float, tuple | Enhanced image |
| `adjust_brightness(image, beta)` | BGR, int | Adjusted image |
| `adjust_contrast(image, alpha)` | BGR, float | Adjusted image |
| `gamma_correction(image, gamma)` | BGR, float | Corrected image |

### Geometry

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `resize(image, width, height)` | BGR, int, int | Resized image |
| `resize_exact(image, width, height)` | BGR, int, int | Resized image |
| `crop(image, x, y, w, h)` | BGR, int×4 | Cropped image |
| `center_crop(image, w, h)` | BGR, int, int | Cropped image |
| `rotate(image, angle, center, scale)` | BGR, float, tuple, float | Rotated image |
| `rotate_90(image)` | BGR | Rotated image |
| `rotate_180(image)` | BGR | Rotated image |
| `rotate_270(image)` | BGR | Rotated image |
| `flip_horizontal(image)` | BGR | Flipped image |
| `flip_vertical(image)` | BGR | Flipped image |
| `flip_both(image)` | BGR | Flipped image |

### Noise

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `add_gaussian_noise(image, mean, sigma)` | BGR, float, float | Noisy image |
| `add_salt_pepper_noise(image, amount)` | BGR, float | Noisy image |
| `add_speckle_noise(image)` | BGR | Noisy image |

---

## ImageFilters

**Module:** `src.filters`

All methods are `@staticmethod`.

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `gaussian_blur(image, kernel)` | BGR, tuple | Blurred image |
| `median_blur(image, kernel_size)` | BGR, int | Blurred image |
| `bilateral_filter(image, d, sigma_color, sigma_space)` | BGR, int, float, float | Filtered image |
| `box_blur(image, kernel)` | BGR, tuple | Blurred image |
| `sharpen(image)` | BGR | Sharpened image |
| `unsharp_mask(image, sigma, strength)` | BGR, float, float | Sharpened image |
| `emboss(image)` | BGR | Embossed image |
| `negative(image)` | BGR | Inverted image |
| `sepia(image)` | BGR | Sepia image |
| `pencil_sketch(image)` | BGR | Grayscale sketch |
| `vignette(image, strength)` | BGR, float | Vignetted image |
| `hdr_effect(image, sigma_s, sigma_r)` | BGR, float, float | HDR image |
| `cartoon(image, num_downsamples, num_bilateral)` | BGR, int, int | Cartoon image |
| `oil_painting(image, size, dyn_ratio)` | BGR, int, int | Oil painting |
| `color_quantize(image, k)` | BGR, int | Quantized image |
| `warm_tone(image, intensity)` | BGR, int | Warm-toned image |
| `cool_tone(image, intensity)` | BGR, int | Cool-toned image |

---

## EdgeDetector

**Module:** `src.edge_detection`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `sobel(image, ksize)` | BGR, int | `(sobel_x, sobel_y, combined)` |
| `laplacian(image)` | BGR | Edge image |
| `canny(image, threshold1, threshold2)` | BGR, int, int | Edge image |
| `prewitt(image)` | BGR | Edge image |
| `roberts(image)` | BGR | Edge image |
| `scharr(image)` | BGR | `(scharr_x, scharr_y, combined)` |
| `edge_overlay(image, edge_color, ...)` | BGR, tuple, ... | BGR with edges |
| `multi_scale_canny(image, scales)` | BGR, list | `dict` of edge images |

---

## Thresholding

**Module:** `src.thresholding`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `binary(image, threshold, max_value)` | BGR, int, int | Binary image |
| `binary_inverse(image, threshold, max_value)` | BGR, int, int | Binary image |
| `truncate(image, threshold)` | BGR, int | Truncated image |
| `to_zero(image, threshold)` | BGR, int | To-zero image |
| `adaptive_mean(image, block_size, c)` | BGR, int, int | Binary image |
| `adaptive_gaussian(image, block_size, c)` | BGR, int, int | Binary image |
| `otsu(image)` | BGR | Binary image |
| `otsu_gaussian(image)` | BGR | Binary image |

---

## Morphology

**Module:** `src.morphology`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `get_kernel(kernel_size)` | tuple | Structuring element |
| `erosion(image, kernel_size, iterations)` | BGR, tuple, int | Eroded image |
| `dilation(image, kernel_size, iterations)` | BGR, tuple, int | Dilated image |
| `opening(image, kernel_size)` | BGR, tuple | Opened image |
| `closing(image, kernel_size)` | BGR, tuple | Closed image |
| `gradient(image, kernel_size)` | BGR, tuple | Gradient image |
| `top_hat(image, kernel_size)` | BGR, tuple | Top-hat image |
| `black_hat(image, kernel_size)` | BGR, tuple | Black-hat image |
| `hit_or_miss(image)` | BGR | Hit-or-miss result |

---

## ContourDetector

**Module:** `src.contour_detection`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `find_contours(image, threshold, mode, method)` | BGR, ... | `(contours, hierarchy)` |
| `draw_contours(image, contours, color, thickness)` | BGR, ... | Annotated image |
| `contour_properties(contours)` | list | `list[dict]` |
| `draw_bounding_boxes(image, contours, ...)` | BGR, ... | Annotated image |
| `draw_rotated_boxes(image, contours, ...)` | BGR, ... | Annotated image |
| `convex_hull(image, contours, ...)` | BGR, ... | Annotated image |
| `approximate_contours(image, contours, ...)` | BGR, ... | Annotated image |
| `min_enclosing_circles(image, contours, ...)` | BGR, ... | Annotated image |
| `full_analysis(image, threshold, min_area)` | BGR, int, int | `(results, properties)` |

---

## ColorAnalysis

**Module:** `src.color_analysis`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `dominant_colors(image, k, max_iter)` | BGR, int, int | `(colors, percentages)` |
| `create_palette(colors, percentages, w, h)` | array, ... | Palette image |
| `to_hsv(image)` | BGR | HSV image |
| `to_lab(image)` | BGR | LAB image |
| `to_hls(image)` | BGR | HLS image |
| `to_ycrcb(image)` | BGR | YCrCb image |
| `split_channels(image)` | BGR | `dict` of channels |
| `color_distribution(image)` | BGR | `dict` of stats |
| `color_temperature(image)` | BGR | `str` |
| `saturation_level(image)` | BGR | `dict` |
| `compare_histograms(img1, img2, method)` | BGR, BGR, str | `float` |
| `full_analysis(image, k)` | BGR, int | `dict` |

---

## ImageMetrics

**Module:** `src.metrics`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `resolution(image)` | any | `dict` |
| `channels(image)` | any | `int` |
| `brightness(image)` | BGR | `float` |
| `contrast(image)` | BGR | `float` |
| `sharpness(image)` | BGR | `float` |
| `entropy(image)` | BGR | `float` |
| `mean(image)` | BGR | `float` |
| `median(image)` | BGR | `float` |
| `variance(image)` | BGR | `float` |
| `minimum(image)` | BGR | `int` |
| `maximum(image)` | BGR | `int` |
| `histogram(image)` | BGR | `ndarray` |
| `rgb_histogram(image)` | BGR | `dict` |
| `psnr(original, processed)` | any, any | `float` |
| `ssim(original, processed)` | any, any | `float` |
| `dominant_colors(image, k)` | BGR, int | `list[dict]` |
| `color_distribution(image)` | BGR | `dict` |
| `image_statistics(image)` | BGR | `dict` |

---

## Visualization

**Module:** `src.visualization`

| Method | Parameters | Description |
|:-------|:-----------|:------------|
| `display_image(title, image, cmap)` | str, ndarray, str | Display image with matplotlib |
| `save_comparison(images, titles, path, cols)` | lists, str, int | Create comparison grid |
| `grayscale_histogram(image, path)` | ndarray, str | Grayscale histogram |
| `rgb_histogram(image, path)` | ndarray, str | RGB histogram |
| `before_after(orig, proc, t1, t2, path)` | ndarrays, strs | Side-by-side comparison |
| `processing_summary(results, path)` | dict, str | Full dashboard |
| `color_palette(colors, pcts, path)` | array, array, str | Color palette visualization |
| `metrics_overlay(image, metrics, path)` | ndarray, dict, str | Image with metrics |

---

## ReportGenerator

**Module:** `src.report_generator`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `generate_text_report(meta, stats, ops, files, name)` | dicts, lists, str | `Path` |
| `generate_markdown_report(...)` | same | `Path` |
| `generate_html_report(...)` | same | `Path` |
| `generate_csv_report(meta, stats, name)` | dicts, str | `Path` |
| `generate_json_report(meta, stats, ops, files, name)` | dicts, lists, str | `Path` |
| `print_summary(meta, stats)` | dicts | `None` |
| `build_file_list(results)` | dict | `list[str]` |

---

## ProcessingPipeline

**Module:** `src.processing_pipeline`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `load_image(path)` | str | `ndarray` |
| `preprocessing()` | — | `None` |
| `filters()` | — | `None` |
| `edge_detection()` | — | `None` |
| `thresholding()` | — | `None` |
| `morphology()` | — | `None` |
| `contours()` | — | `None` |
| `color_analysis()` | — | `None` |
| `metrics()` | — | `None` |
| `process_all()` | — | `None` |
| `process_selected(operations)` | `list[str]` | `None` |
| `get_results()` | — | `dict` |
| `get_metadata()` | — | `dict` |
| `get_statistics()` | — | `dict` |
| `get_timings()` | — | `dict` |
| `get_errors()` | — | `list` |

---

## BatchProcessor

**Module:** `src.batch_processor`

| Method | Parameters | Returns |
|:-------|:-----------|:--------|
| `discover_images(directory)` | str | `list[Path]` |
| `process_single(path, operations, save)` | str, list, bool | `dict \| None` |
| `process_batch(directory, operations, save)` | str, list, bool | `dict` |
| `get_summary()` | — | `dict` |
| `print_summary()` | — | `None` |

---

## Logger

**Module:** `src.logger`

### Functions

| Function | Parameters | Returns |
|:---------|:-----------|:--------|
| `setup_logger(name, level, log_to_file, log_dir)` | str, str, bool, str | `Logger` |
| `get_logger(name)` | str | `Logger` |

### StageTimer

Context manager for timing processing stages:

```python
with StageTimer("Grayscale", logger):
    result = preprocessor.grayscale(image)
```

---

## Utils

**Module:** `src.utils`

| Function | Parameters | Returns |
|:---------|:-----------|:--------|
| `ensure_directory(dir)` | str | `None` |
| `is_supported_file(path)` | str | `bool` |
| `save_image(image, path)` | ndarray, str | `None` |
| `resize_keep_aspect(image, w, h)` | ndarray, int, int | `ndarray` |
| `timestamp()` | — | `str` |
| `timestamp_filename()` | — | `str` |
| `format_file_size(bytes)` | int | `str` |
| `progress_bar(current, total, prefix, length)` | int, int, str, int | `None` |
| `format_table(data, headers, col_width)` | dict/list, tuple, int | `str` |
| `print_header(title)` | str | `None` |
| `print_success(msg)` | str | `None` |
| `print_error(msg)` | str | `None` |
| `print_warning(msg)` | str | `None` |
| `print_info(msg)` | str | `None` |
| `print_stage(name)` | str | `None` |
