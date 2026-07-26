"""
test_image_loader.py
--------------------
Tests for image loading, validation, and metadata.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.image_loader import ImageLoader


class TestImageLoader:

    def test_init(self):
        loader = ImageLoader()
        assert loader.image is None
        assert loader.path is None
        assert loader.metadata == {}

    def test_load_nonexistent_file(self):
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_image.jpg")

    def test_load_unsupported_format(self, tmp_path):
        # Create a dummy .xyz file
        fake = tmp_path / "test.xyz"
        fake.write_text("not an image")

        loader = ImageLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(fake))

    def test_load_valid_image(self, tmp_path, sample_bgr_image):
        # Save a test image
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), sample_bgr_image)

        loader = ImageLoader()
        result = loader.load(str(img_path))

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_metadata_extraction(self, tmp_path, sample_bgr_image):
        img_path = tmp_path / "meta_test.png"
        cv2.imwrite(str(img_path), sample_bgr_image)

        loader = ImageLoader()
        loader.load(str(img_path))

        metadata = loader.get_metadata()

        assert "Filename" in metadata
        assert "Format" in metadata
        assert "Resolution" in metadata
        assert "Width" in metadata
        assert "Height" in metadata
        assert "Channels" in metadata
        assert "Aspect Ratio" in metadata
        assert "File Size (KB)" in metadata
        assert "Data Type" in metadata
        assert "Total Pixels" in metadata
        assert "Megapixels" in metadata
        assert "Bit Depth" in metadata
        assert "Color Space" in metadata

    def test_metadata_values(self, tmp_path, sample_bgr_image):
        img_path = tmp_path / "values_test.png"
        cv2.imwrite(str(img_path), sample_bgr_image)

        loader = ImageLoader()
        loader.load(str(img_path))

        meta = loader.get_metadata()

        assert meta["Width"] == 100
        assert meta["Height"] == 100
        assert meta["Channels"] == 3
        assert meta["Total Pixels"] == 10000
        assert meta["Bit Depth"] == 8

    def test_validate_empty_image(self):
        loader = ImageLoader()
        empty = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError):
            loader._validate_image(empty)

    def test_color_space_detection(self):
        assert ImageLoader._detect_color_space(1) == "Grayscale"
        assert ImageLoader._detect_color_space(3) == "BGR"
        assert ImageLoader._detect_color_space(4) == "BGRA"

    def test_load_grayscale(self, tmp_path, sample_gray_image):
        img_path = tmp_path / "gray.png"
        cv2.imwrite(str(img_path), sample_gray_image)

        loader = ImageLoader()
        result = loader.load(str(img_path))

        # OpenCV may load grayscale PNGs as 3-channel
        assert result is not None
        assert len(result.shape) >= 2
