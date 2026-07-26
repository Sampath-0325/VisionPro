"""
test_morphology.py
------------------
Tests for morphological operations.
"""

import pytest
import numpy as np

from src.morphology import Morphology


class TestMorphology:

    def test_get_kernel(self):
        kernel = Morphology.get_kernel((5, 5))
        assert kernel.shape == (5, 5)
        assert np.all(kernel == 1)

    def test_erosion(self, sample_bgr_image):
        result = Morphology.erosion(sample_bgr_image)
        assert len(result.shape) == 2
        assert result.shape == (100, 100)

    def test_dilation(self, sample_bgr_image):
        result = Morphology.dilation(sample_bgr_image)
        assert len(result.shape) == 2

    def test_erosion_reduces(self, sample_bgr_image):
        eroded = Morphology.erosion(sample_bgr_image)
        dilated = Morphology.dilation(sample_bgr_image)
        # Erosion should generally reduce white areas
        assert np.sum(eroded) <= np.sum(dilated)

    def test_opening(self, sample_bgr_image):
        result = Morphology.opening(sample_bgr_image)
        assert len(result.shape) == 2

    def test_closing(self, sample_bgr_image):
        result = Morphology.closing(sample_bgr_image)
        assert len(result.shape) == 2

    def test_gradient(self, sample_bgr_image):
        result = Morphology.gradient(sample_bgr_image)
        assert len(result.shape) == 2

    def test_top_hat(self, sample_bgr_image):
        result = Morphology.top_hat(sample_bgr_image)
        assert len(result.shape) == 2

    def test_black_hat(self, sample_bgr_image):
        result = Morphology.black_hat(sample_bgr_image)
        assert len(result.shape) == 2

    def test_hit_or_miss(self, sample_bgr_image):
        result = Morphology.hit_or_miss(sample_bgr_image)
        assert len(result.shape) == 2

    def test_custom_kernel_size(self, sample_bgr_image):
        result = Morphology.erosion(
            sample_bgr_image,
            kernel_size=(3, 3)
        )
        assert result.shape == (100, 100)

    def test_multiple_iterations(self, sample_bgr_image):
        result1 = Morphology.erosion(
            sample_bgr_image, iterations=1
        )
        result3 = Morphology.erosion(
            sample_bgr_image, iterations=3
        )
        # More iterations = more erosion
        assert np.sum(result3) <= np.sum(result1)
