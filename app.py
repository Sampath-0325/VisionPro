"""
==============================================================
VisionPro Studio
Advanced Image Processing & Analysis Toolkit
==============================================================

Main Application Entry Point

Usage:
    python app.py --input images/Arches.jpeg
    python app.py --batch images/
    python app.py --input images/Arches.jpeg --operations grayscale,canny
    python app.py --input images/Arches.jpeg --report both --verbose

Author : VisionPro Studio Team
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.config import Config
from src.processing_pipeline import ProcessingPipeline
from src.report_generator import ReportGenerator
from src.visualization import Visualization
from src.batch_processor import BatchProcessor
from src.logger import setup_logger, StageTimer
from src.utils import (
    print_header, print_success, print_error,
    print_warning, print_info, print_stage,
    save_image, format_file_size, progress_bar
)


# ==========================================================
# Argument Parser
# ==========================================================

def create_parser():
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="VisionPro Studio",
        description=(
            "Advanced Image Processing & Analysis Toolkit.\n"
            "Process single images or entire directories with "
            "professional filters, edge detection, thresholding, "
            "morphological operations, and more."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py --input images/Arches.jpeg\n"
            "  python app.py --batch images/\n"
            "  python app.py -i images/Arches.jpeg --operations preprocessing,canny\n"
            "  python app.py -i images/Arches.jpeg --report both --verbose\n"
            "  python app.py -i images/Arches.jpeg --no-viz --format png\n"
        )
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument(
        "-i", "--input",
        type=str,
        help="Path to a single image file."
    )

    input_group.add_argument(
        "-b", "--batch",
        type=str,
        help="Path to a directory of images for batch processing."
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom output directory (default: outputs/)."
    )

    # Operations
    parser.add_argument(
        "--operations",
        type=str,
        default=None,
        help=(
            "Comma-separated list of operations to run. "
            "Options: preprocessing, filters, edge_detection, "
            "thresholding, morphology, contours, color_analysis, metrics. "
            "Default: all operations."
        )
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)."
    )

    # Reports
    parser.add_argument(
        "--report",
        type=str,
        choices=["txt", "md", "html", "csv", "json", "all", "none"],
        default="all",
        help="Report format to generate (default: all)."
    )

    # Visualization
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization/dashboard generation."
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output."
    )

    # List operations
    parser.add_argument(
        "--list-operations",
        action="store_true",
        help="List all available operations and exit."
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"VisionPro Studio v{Config.VERSION}"
    )

    return parser


# ==========================================================
# Save Results
# ==========================================================

def save_results(results, output_format="jpg"):
    """
    Save all processed images into their respective folders.
    """

    total_files = sum(
        len(images) for images in results.values()
    )

    saved = 0

    for folder, images in results.items():

        folder_path = Config.OUTPUT_DIR / folder

        folder_path.mkdir(
            parents=True,
            exist_ok=True
        )

        for filename, image in images.items():

            save_image(
                image,
                folder_path / f"{filename}.{output_format}"
            )

            saved += 1

            progress_bar(
                saved, total_files,
                prefix="  Saving"
            )

    return saved


# ==========================================================
# Generate Reports
# ==========================================================

def generate_reports(
    metadata,
    statistics,
    results,
    report_format="all"
):
    """
    Generate reports in the specified format(s).
    """

    report = ReportGenerator(str(Config.REPORT_DIR))

    output_files = report.build_file_list(results)

    operations_list = [
        "Preprocessing",
        "Filtering",
        "Edge Detection",
        "Thresholding",
        "Morphological Operations",
        "Contour Detection",
        "Color Analysis",
    ]

    generated = []

    if report_format in ("txt", "all"):

        path = report.generate_text_report(
            metadata, statistics,
            operations_list, output_files,
            Config.REPORT_TEXT.name
        )
        generated.append(str(path))

    if report_format in ("md", "all"):

        path = report.generate_markdown_report(
            metadata, statistics,
            operations_list, output_files,
            Config.REPORT_MARKDOWN.name
        )
        generated.append(str(path))

    if report_format in ("html", "all"):

        path = report.generate_html_report(
            metadata, statistics,
            operations_list, output_files,
            Config.REPORT_HTML.name
        )
        generated.append(str(path))

    if report_format in ("csv", "all"):

        path = report.generate_csv_report(
            metadata, statistics,
            Config.REPORT_CSV.name
        )
        generated.append(str(path))

    if report_format in ("json", "all"):

        path = report.generate_json_report(
            metadata, statistics,
            operations_list, output_files,
            Config.REPORT_JSON.name
        )
        generated.append(str(path))

    return report, generated


# ==========================================================
# Generate Visualizations
# ==========================================================

def generate_visualizations(pipeline, results):
    """
    Generate all visualization outputs.
    """

    print_stage("Generating RGB Histogram")

    Visualization.rgb_histogram(
        pipeline.original,
        Config.RGB_HISTOGRAM
    )

    print_success("RGB Histogram saved")

    print_stage("Generating Grayscale Histogram")

    Visualization.grayscale_histogram(
        pipeline.original,
        Config.GRAY_HISTOGRAM
    )

    print_success("Grayscale Histogram saved")

    print_stage("Generating Processing Dashboard")

    Visualization.processing_summary(
        results,
        Config.DASHBOARD
    )

    print_success("Dashboard saved")

    # Before / After
    if "grayscale" in results:

        gray = results["grayscale"].get("grayscale")

        if gray is not None:

            Visualization.before_after(
                pipeline.original,
                gray,
                "Original",
                "Grayscale",
                Config.BEFORE_AFTER
            )

            print_success("Before/After comparison saved")

    # Metrics overlay
    statistics = pipeline.get_statistics()

    if statistics:

        metrics_path = Config.COMPARISON_DIR / "metrics_overlay.png"

        Visualization.metrics_overlay(
            pipeline.original,
            statistics,
            metrics_path
        )

        print_success("Metrics overlay saved")


# ==========================================================
# Process Single Image
# ==========================================================

def process_single(args, logger):
    """
    Process a single image through the full pipeline.
    """

    image_path = Path(
        args.input or Config.DEFAULT_IMAGE
    )

    if not image_path.exists():
        print_error(f"Image not found: {image_path}")
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Parse operations
    operations = None
    if args.operations:
        operations = [
            op.strip()
            for op in args.operations.split(",")
        ]

    # --------------------------------------------------
    # Initialize Pipeline
    # --------------------------------------------------

    pipeline = ProcessingPipeline()

    print_stage("Loading Image")

    pipeline.load_image(image_path)

    metadata = pipeline.get_metadata()

    print_success("Image loaded successfully")

    print()
    for key, value in metadata.items():
        if key == "EXIF":
            continue
        print(f"  {key:<22}: {value}")

    # --------------------------------------------------
    # Process Image
    # --------------------------------------------------

    print_stage("Running Processing Pipeline")

    start_time = datetime.now()

    if operations:
        pipeline.process_selected(operations)
    else:
        pipeline.process_all()

    elapsed = (datetime.now() - start_time).total_seconds()

    print_success(
        f"Processing completed in {elapsed:.2f}s"
    )

    # Print timing breakdown
    timings = pipeline.get_timings()
    if timings:
        print()
        for stage, time_val in timings.items():
            print(f"  {stage:<25}: {time_val:.3f}s")

    # Report errors if any
    errors = pipeline.get_errors()
    if errors:
        print()
        for err in errors:
            print_warning(
                f"  {err['stage']}: {err['error']}"
            )

    # --------------------------------------------------
    # Retrieve Results
    # --------------------------------------------------

    results = pipeline.get_results()

    statistics = pipeline.get_statistics()

    # --------------------------------------------------
    # Save Images
    # --------------------------------------------------

    print_stage("Saving Processed Images")

    saved_count = save_results(results, args.format)

    print_success(f"{saved_count} images saved")

    # --------------------------------------------------
    # Visualizations
    # --------------------------------------------------

    if not args.no_viz:
        generate_visualizations(pipeline, results)

    # --------------------------------------------------
    # Reports
    # --------------------------------------------------

    if args.report != "none":

        print_stage("Generating Reports")

        report, generated = generate_reports(
            metadata, statistics,
            results, args.report
        )

        for path in generated:
            print_success(f"Report: {path}")

        # Console summary
        report.print_summary(metadata, statistics)


# ==========================================================
# Process Batch
# ==========================================================

def process_batch(args, logger):
    """
    Process all images in a directory.
    """

    batch_dir = Path(args.batch)

    if not batch_dir.exists():
        print_error(f"Directory not found: {batch_dir}")
        sys.exit(1)

    if not batch_dir.is_dir():
        print_error(f"Not a directory: {batch_dir}")
        sys.exit(1)

    # Parse operations
    operations = None
    if args.operations:
        operations = [
            op.strip()
            for op in args.operations.split(",")
        ]

    # Initialize batch processor
    output_dir = args.output or str(Config.OUTPUT_DIR)

    processor = BatchProcessor(output_dir=output_dir)

    print_stage(f"Batch processing: {batch_dir}")

    processor.process_batch(
        batch_dir,
        operations=operations,
        save_outputs=True
    )

    processor.print_summary()


# ==========================================================
# Main
# ==========================================================

def main():

    parser = create_parser()
    args = parser.parse_args()

    # List operations and exit
    if args.list_operations:
        print("\nAvailable Operations:")
        print("-" * 40)
        for op in Config.AVAILABLE_OPERATIONS:
            print(f"  • {op}")
        print()
        sys.exit(0)

    # Apply config overrides
    Config.from_args(args)

    if args.output:
        Config.OUTPUT_DIR = Path(args.output)

    # Setup
    print_header(Config.PROJECT_NAME)

    print_info(f"Version {Config.VERSION}")

    # Setup logger
    log_level = "DEBUG" if args.verbose else Config.LOG_LEVEL

    logger = setup_logger(
        level=log_level,
        log_to_file=Config.LOG_TO_FILE,
        log_dir=str(Config.LOG_DIR)
    )

    try:

        # Create directories
        Config.create_project_structure()

        # Route to batch or single
        if args.batch:
            process_batch(args, logger)
        else:
            process_single(args, logger)

        # Final banner
        print("\n")
        print_success("=" * 56)
        print_success("VisionPro Studio Completed Successfully")
        print_success("=" * 56)
        print()

    except KeyboardInterrupt:

        print_warning("\nOperation cancelled by user.")
        sys.exit(130)

    except Exception as error:

        print("\n")
        print_error("=" * 56)
        print_error("APPLICATION ERROR")
        print_error("=" * 56)
        print_error(str(error))

        logger.exception("Unhandled exception")

        sys.exit(1)


# ==========================================================

if __name__ == "__main__":

    main()