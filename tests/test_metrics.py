"""
test_metrics.py
---------------
Tests for image metrics calculations.
"""

import pytest
import numpy as np

from src.metrics import ImageMetrics


class TestImageMetrics:

    def test_resolution(self, sample_bgr_image):
        result = ImageMetrics.resolution(sample_bgr_image)
        assert result["Width"] == 100
        assert result["Height"] == 100
        assert result["Resolution"] == "100 x 100"

    def test_channels_bgr(self, sample_bgr_image):
        assert ImageMetrics.channels(sample_bgr_image) == 3

    def test_channels_gray(self, sample_gray_image):
        assert ImageMetrics.channels(sample_gray_image) == 1

    def test_brightness(self, sample_bgr_image):
        result = ImageMetrics.brightness(sample_bgr_image)
        assert isinstance(result, float)
        assert 0 <= result <= 255

    def test_brightness_white(self, sample_white_image):
        result = ImageMetrics.brightness(sample_white_image)
        assert result == 255.0

    def test_brightness_black(self, sample_black_image):
        result = ImageMetrics.brightness(sample_black_image)
        assert result == 0.0

    def test_contrast(self, sample_bgr_image):
        result = ImageMetrics.contrast(sample_bgr_image)
        assert isinstance(result, float)
        assert result >= 0

    def test_contrast_uniform(self, sample_white_image):
        result = ImageMetrics.contrast(sample_white_image)
        assert result == 0.0

    def test_sharpness(self, sample_bgr_image):
        result = ImageMetrics.sharpness(sample_bgr_image)
        assert isinstance(result, float)
        assert result >= 0

    def test_entropy(self, sample_bgr_image):
        result = ImageMetrics.entropy(sample_bgr_image)
        assert isinstance(result, float)
        assert result >= 0

    def test_mean(self, sample_bgr_image):
        result = ImageMetrics.mean(sample_bgr_image)
        assert isinstance(result, float)

    def test_median(self, sample_bgr_image):
        result = ImageMetrics.median(sample_bgr_image)
        assert isinstance(result, float)

    def test_variance(self, sample_bgr_image):
        result = ImageMetrics.variance(sample_bgr_image)
        assert isinstance(result, float)
        assert result >= 0

    def test_minimum(self, sample_bgr_image):
        result = ImageMetrics.minimum(sample_bgr_image)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_maximum(self, sample_bgr_image):
        result = ImageMetrics.maximum(sample_bgr_image)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_histogram(self, sample_bgr_image):
        result = ImageMetrics.histogram(sample_bgr_image)
        assert result.shape == (256, 1)

    def test_rgb_histogram(self, sample_bgr_image):
        result = ImageMetrics.rgb_histogram(sample_bgr_image)
        assert "blue" in result
        assert "green" in result
        assert "red" in result

    def test_psnr_identical(self, sample_bgr_image):
        result = ImageMetrics.psnr(
            sample_bgr_image, sample_bgr_image
        )
        assert result == float("inf")

    def test_psnr_different(self, sample_bgr_image, sample_white_image):
        result = ImageMetrics.psnr(
            sample_bgr_image, sample_white_image
        )
        assert isinstance(result, float)
        assert result > 0

    def test_ssim_identical(self, sample_bgr_image):
        result = ImageMetrics.ssim(
            sample_bgr_image, sample_bgr_image
        )
        assert abs(result - 1.0) < 0.01

    def test_ssim_different(self, sample_bgr_image, sample_white_image):
        result = ImageMetrics.ssim(
            sample_bgr_image, sample_white_image
        )
        assert isinstance(result, float)
        assert result < 1.0

    def test_dominant_colors(self, sample_bgr_image):
        result = ImageMetrics.dominant_colors(sample_bgr_image, k=3)
        assert len(result) == 3
        assert "BGR" in result[0]
        assert "RGB" in result[0]
        assert "Hex" in result[0]
        assert "Percentage" in result[0]
        # Percentages should sum to ~100
        total = sum(c["Percentage"] for c in result)
        assert abs(total - 100) < 1

    def test_color_distribution(self, sample_bgr_image):
        result = ImageMetrics.color_distribution(sample_bgr_image)
        assert "Blue" in result
        assert "Green" in result
        assert "Red" in result
        for channel in result.values():
            assert "Mean" in channel
            assert "Std" in channel

    def test_image_statistics(self, sample_bgr_image):
        result = ImageMetrics.image_statistics(sample_bgr_image)
        assert "Resolution" in result
        assert "Brightness" in result
        assert "Contrast" in result
        assert "Sharpness" in result
        assert "Entropy" in result
        assert "Color Distribution" in result
