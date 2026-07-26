"""
test_filters.py
---------------
Tests for image filter operations.
"""

import pytest
import numpy as np

from src.filters import ImageFilters


class TestImageFilters:

    def test_gaussian_blur(self, sample_bgr_image):
        result = ImageFilters.gaussian_blur(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_median_blur(self, sample_bgr_image):
        result = ImageFilters.median_blur(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_bilateral_filter(self, sample_bgr_image):
        result = ImageFilters.bilateral_filter(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_box_blur(self, sample_bgr_image):
        result = ImageFilters.box_blur(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_sharpen(self, sample_bgr_image):
        result = ImageFilters.sharpen(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_unsharp_mask(self, sample_bgr_image):
        result = ImageFilters.unsharp_mask(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_emboss(self, sample_bgr_image):
        result = ImageFilters.emboss(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_negative(self, sample_bgr_image):
        result = ImageFilters.negative(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        # Negative of 0 should be 255
        assert result[0, 0, 2] == 255 - sample_bgr_image[0, 0, 2]

    def test_sepia(self, sample_bgr_image):
        result = ImageFilters.sepia(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8

    def test_pencil_sketch(self, sample_bgr_image):
        result = ImageFilters.pencil_sketch(sample_bgr_image)
        assert len(result.shape) == 2  # Grayscale

    def test_vignette(self, sample_bgr_image):
        result = ImageFilters.vignette(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8

    def test_vignette_strength(self, sample_white_image):
        result = ImageFilters.vignette(sample_white_image, strength=0.8)
        # Corners should be darker than center
        center = result[50, 50, 0]
        corner = result[0, 0, 0]
        assert center >= corner

    def test_hdr_effect(self, sample_bgr_image):
        result = ImageFilters.hdr_effect(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_cartoon(self, sample_bgr_image):
        result = ImageFilters.cartoon(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_color_quantize(self, sample_bgr_image):
        result = ImageFilters.color_quantize(sample_bgr_image, k=4)
        assert result.shape == sample_bgr_image.shape
        # Should have at most 4 unique colors
        unique = len(np.unique(result.reshape(-1, 3), axis=0))
        assert unique <= 4

    def test_warm_tone(self, sample_bgr_image):
        result = ImageFilters.warm_tone(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8

    def test_cool_tone(self, sample_bgr_image):
        result = ImageFilters.cool_tone(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8
