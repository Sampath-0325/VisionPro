"""
==============================================================
VisionPro Studio
Batch Processor Module
==============================================================

Process multiple images in a directory with progress
tracking and error isolation.

Author : VisionPro Studio Team
"""

from pathlib import Path
from datetime import datetime

from src.processing_pipeline import ProcessingPipeline
from src.config import Config
from src.logger import get_logger, StageTimer
from src.utils import is_supported_file, save_image


class BatchProcessor:
    """
    Batch image processing with per-image error isolation.
    """

    def __init__(self, output_dir=None):

        self.logger = get_logger()

        self.output_dir = Path(
            output_dir or Config.OUTPUT_DIR
        )

        self.results = []
        self.errors = []
        self.total_processed = 0
        self.total_failed = 0

    # ---------------------------------------------------------
    # Discover Images
    # ---------------------------------------------------------

    @staticmethod
    def discover_images(directory):
        """
        Find all supported image files in a directory.

        Parameters
        ----------
        directory : str or Path
            Directory to scan.

        Returns
        -------
        list[Path]
            Sorted list of image paths.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        images = [
            f for f in sorted(directory.iterdir())
            if f.is_file() and is_supported_file(f)
        ]

        return images

    # ---------------------------------------------------------
    # Process Single Image
    # ---------------------------------------------------------

    def process_single(
        self,
        image_path,
        operations=None,
        save_outputs=True
    ):
        """
        Process a single image with error isolation.

        Returns
        -------
        dict or None
            Processing results, or None on failure.
        """

        image_path = Path(image_path)

        try:

            pipeline = ProcessingPipeline()

            pipeline.load_image(image_path)

            metadata = pipeline.get_metadata()

            if operations:
                pipeline.process_selected(operations)
            else:
                pipeline.process_all()

            results = pipeline.get_results()

            statistics = pipeline.get_statistics()

            # Save outputs
            if save_outputs:

                image_output = (
                    self.output_dir / image_path.stem
                )

                for folder, images in results.items():

                    folder_path = image_output / folder

                    folder_path.mkdir(
                        parents=True, exist_ok=True
                    )

                    for filename, img in images.items():

                        save_image(
                            img,
                            folder_path / f"{filename}.jpg"
                        )

            self.total_processed += 1

            result = {
                "path": str(image_path),
                "filename": image_path.name,
                "metadata": metadata,
                "statistics": statistics,
                "status": "success",
            }

            self.results.append(result)

            return result

        except Exception as error:

            self.total_failed += 1

            error_info = {
                "path": str(image_path),
                "filename": image_path.name,
                "error": str(error),
                "status": "failed",
            }

            self.errors.append(error_info)

            self.logger.error(
                f"Failed to process {image_path.name}: {error}"
            )

            return None

    # ---------------------------------------------------------
    # Process Batch
    # ---------------------------------------------------------

    def process_batch(
        self,
        directory,
        operations=None,
        save_outputs=True
    ):
        """
        Process all images in a directory.

        Parameters
        ----------
        directory : str or Path
            Directory containing images.
        operations : list[str] or None
            Specific operations to run.
        save_outputs : bool
            Whether to save output images.

        Returns
        -------
        dict
            Batch processing summary.
        """

        images = self.discover_images(directory)

        total = len(images)

        if total == 0:
            self.logger.warning(
                f"No supported images found in {directory}"
            )
            return self.get_summary()

        self.logger.info(
            f"Batch processing {total} images from {directory}"
        )

        start_time = datetime.now()

        for index, image_path in enumerate(images, start=1):

            self.logger.info(
                f"[{index}/{total}] Processing: {image_path.name}"
            )

            with StageTimer(
                f"Image {index}/{total}: {image_path.name}",
                self.logger
            ):
                self.process_single(
                    image_path,
                    operations=operations,
                    save_outputs=save_outputs
                )

        elapsed = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            f"Batch complete: {self.total_processed} succeeded, "
            f"{self.total_failed} failed, "
            f"{elapsed:.2f}s total"
        )

        return self.get_summary()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def get_summary(self):
        """
        Get batch processing summary.
        """

        return {
            "total_images": self.total_processed + self.total_failed,
            "successful": self.total_processed,
            "failed": self.total_failed,
            "results": self.results,
            "errors": self.errors,
        }

    # ---------------------------------------------------------
    # Print Summary
    # ---------------------------------------------------------

    def print_summary(self):
        """
        Print a formatted batch processing summary.
        """

        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)

        print(f"\nTotal Images   : {summary['total_images']}")
        print(f"Successful     : {summary['successful']}")
        print(f"Failed         : {summary['failed']}")

        if summary["errors"]:

            print(f"\nFailed Images:")

            for err in summary["errors"]:

                print(f"  [x] {err['filename']}: {err['error']}")

        print("\n" + "=" * 60)
