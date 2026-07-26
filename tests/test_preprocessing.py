"""
test_preprocessing.py
---------------------
Tests for preprocessing operations.
"""

import pytest
import numpy as np
import cv2

from src.preprocessing import Preprocessor


class TestPreprocessor:

    def test_grayscale(self, sample_bgr_image):
        result = Preprocessor.grayscale(sample_bgr_image)
        assert len(result.shape) == 2
        assert result.shape == (100, 100)

    def test_to_hsv(self, sample_bgr_image):
        result = Preprocessor.to_hsv(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8

    def test_to_lab(self, sample_bgr_image):
        result = Preprocessor.to_lab(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_to_rgb(self, sample_bgr_image):
        result = Preprocessor.to_rgb(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_histogram_equalization(self, sample_gray_image):
        result = Preprocessor.histogram_equalization(sample_gray_image)
        assert result.shape == sample_gray_image.shape
        assert result.dtype == np.uint8

    def test_clahe(self, sample_gray_image):
        result = Preprocessor.clahe(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_adjust_brightness(self, sample_bgr_image):
        brighter = Preprocessor.adjust_brightness(sample_bgr_image, beta=50)
        assert brighter.shape == sample_bgr_image.shape
        # Should be brighter overall
        assert np.mean(brighter) >= np.mean(sample_bgr_image)

    def test_adjust_contrast(self, sample_bgr_image):
        result = Preprocessor.adjust_contrast(sample_bgr_image, alpha=1.5)
        assert result.shape == sample_bgr_image.shape

    def test_gamma_correction(self, sample_bgr_image):
        result = Preprocessor.gamma_correction(sample_bgr_image, gamma=0.5)
        assert result.shape == sample_bgr_image.shape

    def test_resize_width(self, sample_bgr_image):
        result = Preprocessor.resize(sample_bgr_image, width=50)
        assert result.shape[1] == 50

    def test_resize_height(self, sample_bgr_image):
        result = Preprocessor.resize(sample_bgr_image, height=50)
        assert result.shape[0] == 50

    def test_resize_both(self, sample_bgr_image):
        result = Preprocessor.resize_exact(sample_bgr_image, 200, 150)
        assert result.shape == (150, 200, 3)

    def test_resize_none(self, sample_bgr_image):
        result = Preprocessor.resize(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_crop(self, sample_bgr_image):
        result = Preprocessor.crop(sample_bgr_image, 10, 10, 50, 50)
        assert result.shape == (50, 50, 3)

    def test_center_crop(self, sample_bgr_image):
        result = Preprocessor.center_crop(sample_bgr_image, 40, 40)
        assert result.shape == (40, 40, 3)

    def test_rotate(self, sample_bgr_image):
        result = Preprocessor.rotate(sample_bgr_image, 45)
        assert result.shape == sample_bgr_image.shape

    def test_rotate_90(self, sample_bgr_image):
        result = Preprocessor.rotate_90(sample_bgr_image)
        assert result.shape[0] == sample_bgr_image.shape[1]
        assert result.shape[1] == sample_bgr_image.shape[0]

    def test_rotate_180(self, sample_bgr_image):
        result = Preprocessor.rotate_180(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_flip_horizontal(self, sample_bgr_image):
        result = Preprocessor.flip_horizontal(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_flip_vertical(self, sample_bgr_image):
        result = Preprocessor.flip_vertical(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_add_gaussian_noise(self, sample_bgr_image):
        result = Preprocessor.add_gaussian_noise(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8
        # Should be different from original
        assert not np.array_equal(result, sample_bgr_image)

    def test_add_salt_pepper_noise(self, sample_bgr_image):
        result = Preprocessor.add_salt_pepper_noise(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert result.dtype == np.uint8

    def test_add_speckle_noise(self, sample_bgr_image):
        result = Preprocessor.add_speckle_noise(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
