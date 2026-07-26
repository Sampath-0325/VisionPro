"""
==============================================================
VisionPro Studio
Configuration Module
==============================================================

Centralized configuration file.

Author : VisionPro Studio Team
"""

import json
from pathlib import Path


class Config:

    # ======================================================
    # PROJECT
    # ======================================================

    PROJECT_NAME = "VisionPro Studio"

    VERSION = "2.0.0"

    AUTHOR = "VisionPro Studio Team"

    DESCRIPTION = (
        "Advanced Image Processing & Analysis Toolkit"
    )

    # ======================================================
    # ROOT
    # ======================================================

    ROOT = Path(__file__).resolve().parent.parent

    # ======================================================
    # DIRECTORIES
    # ======================================================

    IMAGES_DIR = ROOT / "images"

    OUTPUT_DIR = ROOT / "outputs"

    REPORT_DIR = ROOT / "reports"

    ASSETS_DIR = ROOT / "assets"

    DOCS_DIR = ROOT / "docs"

    LOG_DIR = ROOT / "logs"

    # ======================================================
    # OUTPUT FOLDERS
    # ======================================================

    GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"

    HISTOGRAM_DIR = OUTPUT_DIR / "histogram"

    GAUSSIAN_DIR = OUTPUT_DIR / "gaussian"

    MEDIAN_DIR = OUTPUT_DIR / "median"

    BILATERAL_DIR = OUTPUT_DIR / "bilateral"

    SHARPEN_DIR = OUTPUT_DIR / "sharpen"

    SOBEL_DIR = OUTPUT_DIR / "sobel"

    LAPLACIAN_DIR = OUTPUT_DIR / "laplacian"

    CANNY_DIR = OUTPUT_DIR / "canny"

    THRESHOLD_DIR = OUTPUT_DIR / "threshold"

    MORPHOLOGY_DIR = OUTPUT_DIR / "morphology"

    CONTOUR_DIR = OUTPUT_DIR / "contour"

    COLOR_DIR = OUTPUT_DIR / "color"

    COMPARISON_DIR = OUTPUT_DIR / "comparison"

    # ======================================================
    # IMAGE SETTINGS
    # ======================================================

    SUPPORTED_FORMATS = (

        ".jpg",

        ".jpeg",

        ".png",

        ".bmp",

        ".tif",

        ".tiff",

        ".webp",

        ".gif"

    )

    DEFAULT_IMAGE = IMAGES_DIR / "Arches.jpeg"

    # ======================================================
    # FILTER SETTINGS
    # ======================================================

    GAUSSIAN_KERNEL = (5, 5)

    MEDIAN_KERNEL = 5

    BOX_KERNEL = (5, 5)

    BILATERAL_DIAMETER = 9

    BILATERAL_SIGMA_COLOR = 75

    BILATERAL_SIGMA_SPACE = 75

    # ======================================================
    # EDGE SETTINGS
    # ======================================================

    SOBEL_KERNEL = 3

    CANNY_THRESHOLD1 = 100

    CANNY_THRESHOLD2 = 200

    # ======================================================
    # THRESHOLD SETTINGS
    # ======================================================

    THRESHOLD = 127

    MAX_VALUE = 255

    ADAPTIVE_BLOCK_SIZE = 11

    ADAPTIVE_C = 2

    # ======================================================
    # MORPHOLOGY
    # ======================================================

    MORPH_KERNEL = (5, 5)

    MORPH_ITERATIONS = 1

    # ======================================================
    # CONTOUR SETTINGS
    # ======================================================

    CONTOUR_THRESHOLD = 127

    CONTOUR_MIN_AREA = 100

    # ======================================================
    # COLOR ANALYSIS
    # ======================================================

    DOMINANT_COLORS_K = 5

    # ======================================================
    # REPORTS
    # ======================================================

    REPORT_TEXT = REPORT_DIR / "report.txt"

    REPORT_MARKDOWN = REPORT_DIR / "report.md"

    REPORT_HTML = REPORT_DIR / "report.html"

    REPORT_CSV = REPORT_DIR / "metrics.csv"

    REPORT_JSON = REPORT_DIR / "report.json"

    # ======================================================
    # VISUALIZATION
    # ======================================================

    DASHBOARD = COMPARISON_DIR / "dashboard.png"

    RGB_HISTOGRAM = COMPARISON_DIR / "rgb_histogram.png"

    GRAY_HISTOGRAM = COMPARISON_DIR / "gray_histogram.png"

    BEFORE_AFTER = COMPARISON_DIR / "before_after.png"

    COLOR_PALETTE = COMPARISON_DIR / "color_palette.png"

    # ======================================================
    # FIGURE SETTINGS
    # ======================================================

    FIGURE_DPI = 300

    FIGURE_COLUMNS = 4

    FIGURE_SIZE = (16, 12)

    # ======================================================
    # GUI SETTINGS
    # ======================================================

    WINDOW_WIDTH = 1400

    WINDOW_HEIGHT = 850

    SIDEBAR_WIDTH = 320

    PREVIEW_WIDTH = 500

    PREVIEW_HEIGHT = 500

    # ======================================================
    # LOGGING
    # ======================================================

    ENABLE_LOGGING = True

    LOG_LEVEL = "INFO"

    LOG_TO_FILE = True

    # ======================================================
    # AVAILABLE OPERATIONS
    # ======================================================

    AVAILABLE_OPERATIONS = [
        "preprocessing",
        "filters",
        "edge_detection",
        "thresholding",
        "morphology",
        "contours",
        "color_analysis",
        "metrics",
    ]

    # ======================================================
    # CREATE DIRECTORIES
    # ======================================================

    @classmethod
    def create_project_structure(cls):

        folders = [

            cls.IMAGES_DIR,

            cls.OUTPUT_DIR,

            cls.REPORT_DIR,

            cls.ASSETS_DIR,

            cls.DOCS_DIR,

            cls.LOG_DIR,

            cls.GRAYSCALE_DIR,

            cls.HISTOGRAM_DIR,

            cls.GAUSSIAN_DIR,

            cls.MEDIAN_DIR,

            cls.BILATERAL_DIR,

            cls.SHARPEN_DIR,

            cls.SOBEL_DIR,

            cls.LAPLACIAN_DIR,

            cls.CANNY_DIR,

            cls.THRESHOLD_DIR,

            cls.MORPHOLOGY_DIR,

            cls.CONTOUR_DIR,

            cls.COLOR_DIR,

            cls.COMPARISON_DIR

        ]

        for folder in folders:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    # ======================================================
    # FROM CLI ARGUMENTS
    # ======================================================

    @classmethod
    def from_args(cls, args):
        """
        Override config values from CLI arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed CLI arguments.
        """

        if hasattr(args, "output") and args.output:
            cls.OUTPUT_DIR = Path(args.output)

        if hasattr(args, "input") and args.input:
            cls.DEFAULT_IMAGE = Path(args.input)

        if hasattr(args, "log_level") and args.log_level:
            cls.LOG_LEVEL = args.log_level.upper()

        if hasattr(args, "verbose") and args.verbose:
            cls.LOG_LEVEL = "DEBUG"

        return cls

    # ======================================================
    # CONFIG VALIDATION
    # ======================================================

    @classmethod
    def validate(cls):
        """
        Validate the current configuration.

        Returns
        -------
        list[str]
            List of validation warning messages.
        """

        warnings = []

        if not cls.IMAGES_DIR.exists():
            warnings.append(
                f"Images directory not found: {cls.IMAGES_DIR}"
            )

        if (
            cls.DEFAULT_IMAGE
            and not cls.DEFAULT_IMAGE.exists()
        ):
            warnings.append(
                f"Default image not found: {cls.DEFAULT_IMAGE}"
            )

        return warnings

    # ======================================================
    # JSON CONFIG LOADING
    # ======================================================

    @classmethod
    def load_from_json(cls, json_path):
        """
        Load configuration overrides from a JSON file.

        Parameters
        ----------
        json_path : str or Path
            Path to JSON configuration file.
        """

        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Apply overrides
        for key, value in config_data.items():

            key_upper = key.upper()

            if hasattr(cls, key_upper):

                current = getattr(cls, key_upper)

                # Convert path strings to Path objects
                if isinstance(current, Path):
                    value = Path(value)

                # Convert list to tuple for tuples
                elif isinstance(current, tuple):
                    value = tuple(value)

                setattr(cls, key_upper, value)

    # ======================================================
    # EXPORT CONFIG
    # ======================================================

    @classmethod
    def to_dict(cls):
        """
        Export current configuration as a dictionary.
        """

        config = {}

        for key in dir(cls):

            if key.startswith("_"):
                continue

            if callable(getattr(cls, key)):
                continue

            value = getattr(cls, key)

            if isinstance(value, Path):
                value = str(value)

            config[key] = value

        return config