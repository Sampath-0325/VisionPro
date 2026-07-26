"""
test_thresholding.py
--------------------
Tests for thresholding operations.
"""

import pytest
import numpy as np

from src.thresholding import Thresholding


class TestThresholding:

    def test_binary(self, sample_bgr_image):
        result = Thresholding.binary(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_binary_inverse(self, sample_bgr_image):
        result = Thresholding.binary_inverse(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_binary_vs_inverse(self, sample_bgr_image):
        binary = Thresholding.binary(sample_bgr_image)
        inverse = Thresholding.binary_inverse(sample_bgr_image)
        # They should be complements
        combined = cv2.add(binary, inverse)
        assert np.all(combined == 255)

    def test_truncate(self, sample_bgr_image):
        result = Thresholding.truncate(sample_bgr_image)
        assert len(result.shape) == 2
        assert np.max(result) <= 255

    def test_to_zero(self, sample_bgr_image):
        result = Thresholding.to_zero(sample_bgr_image)
        assert len(result.shape) == 2

    def test_adaptive_mean(self, sample_bgr_image):
        result = Thresholding.adaptive_mean(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_adaptive_gaussian(self, sample_bgr_image):
        result = Thresholding.adaptive_gaussian(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_otsu(self, sample_bgr_image):
        result = Thresholding.otsu(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_otsu_gaussian(self, sample_bgr_image):
        result = Thresholding.otsu_gaussian(sample_bgr_image)
        assert len(result.shape) == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_custom_threshold(self, sample_bgr_image):
        result = Thresholding.binary(
            sample_bgr_image, threshold=200
        )
        assert len(result.shape) == 2


# Need cv2 for the complement test
import cv2
