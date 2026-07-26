"""
test_utils.py
-------------
Tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils import (
    is_supported_file,
    ensure_directory,
    format_file_size,
    timestamp,
    timestamp_filename,
    resize_keep_aspect,
    format_table,
    SUPPORTED_EXTENSIONS,
)


class TestUtils:

    def test_supported_extensions_includes_common(self):
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS
        assert ".gif" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS

    def test_is_supported_jpg(self):
        assert is_supported_file("test.jpg") is True

    def test_is_supported_png(self):
        assert is_supported_file("test.png") is True

    def test_is_supported_gif(self):
        assert is_supported_file("test.gif") is True

    def test_is_not_supported(self):
        assert is_supported_file("test.xyz") is False
        assert is_supported_file("test.txt") is False
        assert is_supported_file("test.pdf") is False

    def test_is_supported_case_insensitive(self):
        assert is_supported_file("test.JPG") is True
        assert is_supported_file("test.PNG") is True

    def test_ensure_directory(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        ensure_directory(new_dir)
        assert new_dir.exists()

    def test_format_file_size_bytes(self):
        assert format_file_size(500) == "500 B"

    def test_format_file_size_kb(self):
        result = format_file_size(2048)
        assert "KB" in result

    def test_format_file_size_mb(self):
        result = format_file_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_file_size_gb(self):
        result = format_file_size(2 * 1024 ** 3)
        assert "GB" in result

    def test_timestamp_format(self):
        result = timestamp()
        assert len(result) == 19  # YYYY-MM-DD HH:MM:SS

    def test_timestamp_filename_no_colons(self):
        result = timestamp_filename()
        assert ":" not in result

    def test_resize_keep_aspect_width(self, sample_bgr_image):
        result = resize_keep_aspect(sample_bgr_image, width=50)
        assert result.shape[1] == 50

    def test_resize_keep_aspect_height(self, sample_bgr_image):
        result = resize_keep_aspect(sample_bgr_image, height=50)
        assert result.shape[0] == 50

    def test_resize_keep_aspect_none(self, sample_bgr_image):
        result = resize_keep_aspect(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_format_table_dict(self):
        data = {"key1": "value1", "key2": "value2"}
        result = format_table(data)
        assert "key1" in result
        assert "value1" in result

    def test_format_table_list(self):
        data = [("a", "b"), ("c", "d")]
        result = format_table(data)
        assert "a" in result

    def test_format_table_headers(self):
        data = {"key": "value"}
        result = format_table(data, headers=("Name", "Value"))
        assert "Name" in result
