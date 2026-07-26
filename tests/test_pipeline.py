"""
test_pipeline.py
----------------
Tests for the processing pipeline.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.processing_pipeline import ProcessingPipeline


class TestProcessingPipeline:

    @pytest.fixture
    def pipeline_with_image(self, tmp_path, sample_bgr_image):
        """Create a pipeline loaded with a test image."""
        img_path = tmp_path / "pipeline_test.jpg"
        cv2.imwrite(str(img_path), sample_bgr_image)

        pipeline = ProcessingPipeline()
        pipeline.load_image(str(img_path))
        return pipeline

    def test_init(self):
        pipeline = ProcessingPipeline()
        assert pipeline.original is None
        assert pipeline.metadata == {}
        assert pipeline.statistics == {}
        assert pipeline.results == {}

    def test_load_image(self, tmp_path, sample_bgr_image):
        img_path = tmp_path / "load_test.jpg"
        cv2.imwrite(str(img_path), sample_bgr_image)

        pipeline = ProcessingPipeline()
        result = pipeline.load_image(str(img_path))

        assert result is not None
        assert pipeline.original is not None
        assert len(pipeline.metadata) > 0

    def test_preprocessing(self, pipeline_with_image):
        pipeline_with_image.preprocessing()
        results = pipeline_with_image.get_results()
        assert "grayscale" in results
        assert "histogram" in results

    def test_filters(self, pipeline_with_image):
        pipeline_with_image.filters()
        results = pipeline_with_image.get_results()
        assert "gaussian" in results
        assert "sharpen" in results

    def test_edge_detection(self, pipeline_with_image):
        pipeline_with_image.edge_detection()
        results = pipeline_with_image.get_results()
        assert "sobel" in results
        assert "laplacian" in results
        assert "canny" in results

    def test_thresholding(self, pipeline_with_image):
        pipeline_with_image.thresholding()
        results = pipeline_with_image.get_results()
        assert "threshold" in results

    def test_morphology(self, pipeline_with_image):
        pipeline_with_image.morphology()
        results = pipeline_with_image.get_results()
        assert "morphology" in results

    def test_contours(self, pipeline_with_image):
        pipeline_with_image.contours()
        results = pipeline_with_image.get_results()
        assert "contour" in results

    def test_color_analysis(self, pipeline_with_image):
        pipeline_with_image.color_analysis()
        results = pipeline_with_image.get_results()
        assert "color" in results

    def test_metrics(self, pipeline_with_image):
        pipeline_with_image.metrics()
        stats = pipeline_with_image.get_statistics()
        assert "Brightness" in stats
        assert "Contrast" in stats

    def test_process_all(self, pipeline_with_image):
        pipeline_with_image.process_all()
        results = pipeline_with_image.get_results()
        assert len(results) > 0
        stats = pipeline_with_image.get_statistics()
        assert len(stats) > 0
        timings = pipeline_with_image.get_timings()
        assert len(timings) > 0

    def test_process_selected(self, pipeline_with_image):
        pipeline_with_image.process_selected(
            ["preprocessing", "metrics"]
        )
        results = pipeline_with_image.get_results()
        assert "grayscale" in results
        stats = pipeline_with_image.get_statistics()
        assert "Brightness" in stats

    def test_process_selected_unknown(self, pipeline_with_image):
        """Unknown operations should be skipped gracefully."""
        pipeline_with_image.process_selected(
            ["preprocessing", "unknown_op"]
        )
        results = pipeline_with_image.get_results()
        assert "grayscale" in results

    def test_get_timings(self, pipeline_with_image):
        pipeline_with_image.process_all()
        timings = pipeline_with_image.get_timings()
        assert isinstance(timings, dict)
        assert all(isinstance(v, float) for v in timings.values())

    def test_get_errors(self, pipeline_with_image):
        errors = pipeline_with_image.get_errors()
        assert isinstance(errors, list)
