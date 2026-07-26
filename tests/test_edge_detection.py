"""
test_edge_detection.py
----------------------
Tests for edge detection algorithms.
"""

import pytest
import numpy as np

from src.edge_detection import EdgeDetector


class TestEdgeDetector:

    def test_sobel(self, sample_bgr_image):
        sx, sy, combined = EdgeDetector.sobel(sample_bgr_image)
        assert len(sx.shape) == 2
        assert len(sy.shape) == 2
        assert len(combined.shape) == 2
        assert sx.shape == (100, 100)

    def test_laplacian(self, sample_bgr_image):
        result = EdgeDetector.laplacian(sample_bgr_image)
        assert len(result.shape) == 2
        assert result.shape == (100, 100)

    def test_canny(self, sample_bgr_image):
        result = EdgeDetector.canny(sample_bgr_image)
        assert len(result.shape) == 2
        assert result.dtype == np.uint8

    def test_canny_custom_thresholds(self, sample_bgr_image):
        result = EdgeDetector.canny(
            sample_bgr_image,
            threshold1=50,
            threshold2=150
        )
        assert result is not None

    def test_prewitt(self, sample_bgr_image):
        result = EdgeDetector.prewitt(sample_bgr_image)
        assert len(result.shape) == 2

    def test_roberts(self, sample_bgr_image):
        result = EdgeDetector.roberts(sample_bgr_image)
        assert len(result.shape) == 2

    def test_scharr(self, sample_bgr_image):
        sx, sy, combined = EdgeDetector.scharr(sample_bgr_image)
        assert len(sx.shape) == 2
        assert len(sy.shape) == 2
        assert combined.shape == (100, 100)

    def test_edge_overlay(self, sample_bgr_image):
        result = EdgeDetector.edge_overlay(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        assert len(result.shape) == 3  # Still BGR

    def test_edge_overlay_custom_color(self, sample_bgr_image):
        result = EdgeDetector.edge_overlay(
            sample_bgr_image,
            edge_color=(255, 0, 0)
        )
        assert result.shape == sample_bgr_image.shape

    def test_multi_scale_canny(self, sample_bgr_image):
        results = EdgeDetector.multi_scale_canny(sample_bgr_image)
        assert "high_sensitivity" in results
        assert "medium" in results
        assert "low_sensitivity" in results
        for key, edges in results.items():
            assert len(edges.shape) == 2

    def test_edges_detect_strong_borders(self):
        """Verify edges are detected at sharp boundaries."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, 50:] = 255  # Sharp vertical edge

        edges = EdgeDetector.canny(image, 50, 150)
        # There should be non-zero pixels near column 50
        edge_region = edges[:, 48:52]
        assert np.sum(edge_region) > 0
