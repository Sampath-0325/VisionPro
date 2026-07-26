"""
==============================================================
VisionPro Studio
Processing Pipeline
==============================================================

Central processing engine.

Coordinates every module of VisionPro Studio with
selective processing, timing, and error isolation.

Author : VisionPro Studio Team
"""

from pathlib import Path
from datetime import datetime

from src.image_loader import ImageLoader
from src.preprocessing import Preprocessor
from src.filters import ImageFilters
from src.edge_detection import EdgeDetector
from src.thresholding import Thresholding
from src.morphology import Morphology
from src.metrics import ImageMetrics
from src.contour_detection import ContourDetector
from src.color_analysis import ColorAnalysis
from src.logger import get_logger, StageTimer


class ProcessingPipeline:

    def __init__(self):

        self.loader = ImageLoader()

        self.logger = get_logger()

        self.original = None

        self.metadata = {}

        self.statistics = {}

        self.results = {}

        self.timings = {}

        self.errors = []

    # ---------------------------------------------------------
    # Load Image
    # ---------------------------------------------------------

    def load_image(self, image_path):

        self.original = self.loader.load(
            Path(image_path)
        )

        self.metadata = self.loader.get_metadata()

        return self.original

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    def preprocessing(self):

        gray = Preprocessor.grayscale(
            self.original
        )

        self.results["grayscale"] = {

            "grayscale": gray

        }

        self.results["histogram"] = {

            "histogram_equalization":
                Preprocessor.histogram_equalization(gray),

            "clahe":
                Preprocessor.clahe(gray),

            "brightness":
                Preprocessor.adjust_brightness(
                    self.original
                ),

            "contrast":
                Preprocessor.adjust_contrast(
                    self.original
                ),

            "gamma":
                Preprocessor.gamma_correction(
                    self.original
                )

        }

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------

    def filters(self):

        self.results["gaussian"] = {

            "gaussian":
                ImageFilters.gaussian_blur(
                    self.original
                ),

            "box_blur":
                ImageFilters.box_blur(
                    self.original
                )

        }

        self.results["median"] = {

            "median":
                ImageFilters.median_blur(
                    self.original
                )

        }

        self.results["bilateral"] = {

            "bilateral":
                ImageFilters.bilateral_filter(
                    self.original
                )

        }

        self.results["sharpen"] = {

            "sharpen":
                ImageFilters.sharpen(
                    self.original
                ),

            "unsharp_mask":
                ImageFilters.unsharp_mask(
                    self.original
                ),

            "emboss":
                ImageFilters.emboss(
                    self.original
                ),

            "negative":
                ImageFilters.negative(
                    self.original
                ),

            "sepia":
                ImageFilters.sepia(
                    self.original
                ),

            "sketch":
                ImageFilters.pencil_sketch(
                    self.original
                ),

            "vignette":
                ImageFilters.vignette(
                    self.original
                ),

            "cartoon":
                ImageFilters.cartoon(
                    self.original
                ),

            "warm_tone":
                ImageFilters.warm_tone(
                    self.original
                ),

            "cool_tone":
                ImageFilters.cool_tone(
                    self.original
                ),

            "color_quantize":
                ImageFilters.color_quantize(
                    self.original
                ),

        }

    # ---------------------------------------------------------
    # Edge Detection
    # ---------------------------------------------------------

    def edge_detection(self):

        sx, sy, sobel = EdgeDetector.sobel(
            self.original
        )

        self.results["sobel"] = {

            "sobel_x": sx,

            "sobel_y": sy,

            "sobel": sobel,

            "prewitt":
                EdgeDetector.prewitt(
                    self.original
                ),

            "roberts":
                EdgeDetector.roberts(
                    self.original
                )

        }

        # Scharr
        scx, scy, scharr = EdgeDetector.scharr(
            self.original
        )

        self.results["sobel"]["scharr"] = scharr

        self.results["laplacian"] = {

            "laplacian":
                EdgeDetector.laplacian(
                    self.original
                )

        }

        self.results["canny"] = {

            "canny":
                EdgeDetector.canny(
                    self.original
                ),

            "edge_overlay":
                EdgeDetector.edge_overlay(
                    self.original
                ),

        }

    # ---------------------------------------------------------
    # Thresholding
    # ---------------------------------------------------------

    def thresholding(self):

        self.results["threshold"] = {

            "binary":
                Thresholding.binary(
                    self.original
                ),

            "binary_inverse":
                Thresholding.binary_inverse(
                    self.original
                ),

            "truncate":
                Thresholding.truncate(
                    self.original
                ),

            "to_zero":
                Thresholding.to_zero(
                    self.original
                ),

            "adaptive_mean":
                Thresholding.adaptive_mean(
                    self.original
                ),

            "adaptive_gaussian":
                Thresholding.adaptive_gaussian(
                    self.original
                ),

            "otsu":
                Thresholding.otsu(
                    self.original
                ),

            "otsu_gaussian":
                Thresholding.otsu_gaussian(
                    self.original
                )

        }

    # ---------------------------------------------------------
    # Morphology
    # ---------------------------------------------------------

    def morphology(self):

        self.results["morphology"] = {

            "erosion":
                Morphology.erosion(
                    self.original
                ),

            "dilation":
                Morphology.dilation(
                    self.original
                ),

            "opening":
                Morphology.opening(
                    self.original
                ),

            "closing":
                Morphology.closing(
                    self.original
                ),

            "gradient":
                Morphology.gradient(
                    self.original
                ),

            "top_hat":
                Morphology.top_hat(
                    self.original
                ),

            "black_hat":
                Morphology.black_hat(
                    self.original
                ),

            "hit_or_miss":
                Morphology.hit_or_miss(
                    self.original
                )

        }

    # ---------------------------------------------------------
    # Contour Detection
    # ---------------------------------------------------------

    def contours(self):

        contour_results, properties = (
            ContourDetector.full_analysis(self.original)
        )

        self.results["contour"] = contour_results

        self.statistics["Contour Count"] = len(properties)

        if properties:
            self.statistics["Contour Properties"] = properties

    # ---------------------------------------------------------
    # Color Analysis
    # ---------------------------------------------------------

    def color_analysis(self):

        analysis = ColorAnalysis.full_analysis(
            self.original
        )

        self.results["color"] = {
            "palette": analysis["palette"],
        }

        self.statistics["Color Temperature"] = (
            analysis["temperature"]
        )

        self.statistics["Saturation"] = (
            analysis["saturation"]
        )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def metrics(self):

        self.statistics.update(
            ImageMetrics.image_statistics(
                self.original
            )
        )

    # ---------------------------------------------------------
    # Process All
    # ---------------------------------------------------------

    def process_all(self):
        """
        Run all processing stages with timing and
        error isolation.
        """

        stages = [
            ("Preprocessing", self.preprocessing),
            ("Filters", self.filters),
            ("Edge Detection", self.edge_detection),
            ("Thresholding", self.thresholding),
            ("Morphology", self.morphology),
            ("Contour Detection", self.contours),
            ("Color Analysis", self.color_analysis),
            ("Metrics", self.metrics),
        ]

        for stage_name, stage_func in stages:

            start = datetime.now()

            try:
                with StageTimer(stage_name, self.logger):
                    stage_func()

            except Exception as error:

                self.logger.error(
                    f"Stage '{stage_name}' failed: {error}"
                )

                self.errors.append({
                    "stage": stage_name,
                    "error": str(error)
                })

            elapsed = (datetime.now() - start).total_seconds()

            self.timings[stage_name] = round(elapsed, 3)

    # ---------------------------------------------------------
    # Process Selected Operations
    # ---------------------------------------------------------

    def process_selected(self, operations):
        """
        Run only the specified operations.

        Parameters
        ----------
        operations : list[str]
            List of operation names to run.
            Valid names: preprocessing, filters,
            edge_detection, thresholding, morphology,
            contours, color_analysis, metrics
        """

        stage_map = {
            "preprocessing": ("Preprocessing", self.preprocessing),
            "filters": ("Filters", self.filters),
            "edge_detection": ("Edge Detection", self.edge_detection),
            "thresholding": ("Thresholding", self.thresholding),
            "morphology": ("Morphology", self.morphology),
            "contours": ("Contour Detection", self.contours),
            "color_analysis": ("Color Analysis", self.color_analysis),
            "metrics": ("Metrics", self.metrics),
        }

        for op in operations:

            op_lower = op.lower().strip()

            if op_lower not in stage_map:

                self.logger.warning(
                    f"Unknown operation: '{op}' — skipping"
                )
                continue

            stage_name, stage_func = stage_map[op_lower]

            start = datetime.now()

            try:
                with StageTimer(stage_name, self.logger):
                    stage_func()

            except Exception as error:

                self.logger.error(
                    f"Stage '{stage_name}' failed: {error}"
                )

                self.errors.append({
                    "stage": stage_name,
                    "error": str(error)
                })

            elapsed = (datetime.now() - start).total_seconds()

            self.timings[stage_name] = round(elapsed, 3)

    # ---------------------------------------------------------
    # Getters
    # ---------------------------------------------------------

    def get_results(self):

        return self.results

    # ---------------------------------------------------------

    def get_metadata(self):

        return self.metadata

    # ---------------------------------------------------------

    def get_statistics(self):

        return self.statistics

    # ---------------------------------------------------------

    def get_timings(self):

        return self.timings

    # ---------------------------------------------------------

    def get_errors(self):

        return self.errors