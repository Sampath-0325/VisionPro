"""
image_loader.py
---------------
Handles image loading, validation, and metadata extraction.
Supports standard formats plus GIF (first frame).
"""

from pathlib import Path
import cv2
import os
import numpy as np

from src.utils import is_supported_file

try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


class ImageLoader:

    def __init__(self):
        self.image = None
        self.path = None
        self.metadata = {}

    # ---------------------------------------------------------
    # Load Image
    # ---------------------------------------------------------

    def load(self, filepath):
        """
        Load an image from disk.

        Supports standard OpenCV formats plus GIF files.
        GIF files are loaded using Pillow and the first
        frame is converted to a BGR numpy array.
        """

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Image not found: {filepath}"
            )

        if not is_supported_file(filepath):
            raise ValueError(
                f"Unsupported image format: {filepath.suffix}"
            )

        # Handle GIF files via Pillow
        if filepath.suffix.lower() == ".gif":
            image = self._load_gif(filepath)
        else:
            image = cv2.imread(str(filepath))

        if image is None:
            raise ValueError(
                f"Unable to read image: {filepath}"
            )

        # Validate image data
        self._validate_image(image)

        self.image = image
        self.path = filepath

        self._extract_metadata()

        return self.image

    # ---------------------------------------------------------
    # GIF Loading
    # ---------------------------------------------------------

    @staticmethod
    def _load_gif(filepath):
        """
        Load GIF using Pillow and convert first frame to BGR.
        """

        if not HAS_PILLOW:
            raise ImportError(
                "Pillow is required for GIF support. "
                "Install with: pip install pillow"
            )

        pil_image = PILImage.open(filepath)

        # Convert to RGB (GIFs may be palette-mode)
        pil_image = pil_image.convert("RGB")

        # Convert to numpy array
        rgb_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    # ---------------------------------------------------------
    # Image Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_image(image):
        """
        Validate that image data is not corrupt.
        """

        if not isinstance(image, np.ndarray):
            raise ValueError("Image is not a valid numpy array.")

        if image.size == 0:
            raise ValueError("Image is empty (0 pixels).")

        if len(image.shape) < 2:
            raise ValueError("Image has invalid dimensions.")

        height, width = image.shape[:2]

        if height < 1 or width < 1:
            raise ValueError(
                f"Image has invalid size: {width}x{height}"
            )

    # ---------------------------------------------------------
    # Metadata Extraction
    # ---------------------------------------------------------

    def _extract_metadata(self):
        """
        Extract comprehensive image metadata.
        """

        height, width = self.image.shape[:2]

        channels = (
            1 if len(self.image.shape) == 2
            else self.image.shape[2]
        )

        filesize = os.path.getsize(self.path)

        # Detect color space
        color_space = self._detect_color_space(channels)

        # Build metadata
        self.metadata = {

            "Filename": self.path.name,

            "Format": self.path.suffix.upper(),

            "Resolution": f"{width} x {height}",

            "Width": width,

            "Height": height,

            "Channels": channels,

            "Color Space": color_space,

            "Aspect Ratio": round(width / height, 2),

            "File Size (KB)": round(filesize / 1024, 2),

            "Data Type": str(self.image.dtype),

            "Bit Depth": self.image.dtype.itemsize * 8,

            "Total Pixels": width * height,

            "Megapixels": round(
                (width * height) / 1_000_000, 2
            ),

        }

        # Add EXIF if available
        exif = self._extract_exif()

        if exif:
            self.metadata["EXIF"] = exif

    # ---------------------------------------------------------
    # Color Space Detection
    # ---------------------------------------------------------

    @staticmethod
    def _detect_color_space(channels):
        """
        Infer color space from channel count.
        """

        space_map = {
            1: "Grayscale",
            3: "BGR",
            4: "BGRA",
        }

        return space_map.get(channels, f"Unknown ({channels}ch)")

    # ---------------------------------------------------------
    # EXIF Extraction
    # ---------------------------------------------------------

    def _extract_exif(self):
        """
        Extract EXIF metadata using Pillow.
        Returns None if Pillow is not available or
        no EXIF data exists.
        """

        if not HAS_PILLOW:
            return None

        try:

            pil_image = PILImage.open(self.path)

            exif_data = pil_image.getexif()

            if not exif_data:
                return None

            exif = {}

            # Tags we care about
            desired_tags = {
                "Make", "Model", "DateTime",
                "ExposureTime", "FNumber",
                "ISOSpeedRatings", "FocalLength",
                "ImageWidth", "ImageLength",
                "Software", "Orientation",
            }

            for tag_id, value in exif_data.items():

                tag_name = TAGS.get(tag_id, str(tag_id))

                if tag_name in desired_tags:
                    exif[tag_name] = str(value)

            return exif if exif else None

        except Exception:
            return None

    # ---------------------------------------------------------
    # Public Methods
    # ---------------------------------------------------------

    def get_metadata(self):
        """Return extracted metadata."""
        return self.metadata

    def display_metadata(self):
        """Print metadata to console."""

        print("\nImage Information\n")

        for key, value in self.metadata.items():

            if key == "EXIF":
                print(f"\n{'EXIF Data':<22}")
                for ek, ev in value.items():
                    print(f"  {ek:<20}: {ev}")
            else:
                print(f"{key:<22}: {value}")