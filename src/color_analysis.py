"""
==============================================================
VisionPro Studio
Color Analysis Module
==============================================================

Dominant color extraction, palette generation, and
color space conversions.

Author : VisionPro Studio Team
"""

import cv2
import numpy as np


class ColorAnalysis:
    """
    Color analysis and extraction operations.
    """

    # ---------------------------------------------------------
    # Dominant Colors (K-Means)
    # ---------------------------------------------------------

    @staticmethod
    def dominant_colors(image, k=5, max_iter=100):
        """
        Extract dominant colors using K-Means clustering.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image.
        k : int
            Number of dominant colors.
        max_iter : int
            Maximum K-Means iterations.

        Returns
        -------
        tuple
            (colors_bgr, percentages)
            - colors_bgr: array of dominant BGR colors
            - percentages: percentage of each color
        """

        # Reshape to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # K-Means criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            max_iter,
            0.2
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Calculate percentages
        _, counts = np.unique(labels, return_counts=True)

        percentages = counts / counts.sum() * 100

        # Sort by frequency (most dominant first)
        sort_idx = np.argsort(-percentages)

        centers = centers[sort_idx].astype(np.uint8)

        percentages = percentages[sort_idx]

        return centers, np.round(percentages, 2)

    # ---------------------------------------------------------
    # Color Palette Image
    # ---------------------------------------------------------

    @staticmethod
    def create_palette(
        colors,
        percentages=None,
        width=500,
        height=80
    ):
        """
        Create a color palette image from dominant colors.

        Parameters
        ----------
        colors : numpy.ndarray
            Array of BGR color values.
        percentages : numpy.ndarray or None
            Proportional widths. Equal if None.
        width : int
            Total palette width.
        height : int
            Palette height.

        Returns
        -------
        numpy.ndarray
            Palette image (BGR).
        """

        palette = np.zeros(
            (height, width, 3), dtype=np.uint8
        )

        if percentages is None:
            percentages = np.ones(len(colors)) / len(colors) * 100

        start = 0

        for color, pct in zip(colors, percentages):

            end = start + int(width * pct / 100)

            palette[:, start:end] = color

            start = end

        # Fill remaining pixels
        if start < width:
            palette[:, start:] = colors[-1]

        return palette

    # ---------------------------------------------------------
    # Color Space Conversions
    # ---------------------------------------------------------

    @staticmethod
    def to_hsv(image):
        """Convert BGR to HSV."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def to_lab(image):
        """Convert BGR to LAB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def to_hls(image):
        """Convert BGR to HLS."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    @staticmethod
    def to_ycrcb(image):
        """Convert BGR to YCrCb."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # ---------------------------------------------------------
    # Channel Splitting
    # ---------------------------------------------------------

    @staticmethod
    def split_channels(image):
        """
        Split image into individual BGR channels.

        Returns
        -------
        dict
            {"blue": ndarray, "green": ndarray, "red": ndarray}
        """

        b, g, r = cv2.split(image)

        return {
            "blue": b,
            "green": g,
            "red": r
        }

    # ---------------------------------------------------------
    # Color Distribution
    # ---------------------------------------------------------

    @staticmethod
    def color_distribution(image):
        """
        Analyze color distribution across channels.

        Returns
        -------
        dict
            Mean, std, min, max for each channel.
        """

        channels = {"Blue": 0, "Green": 1, "Red": 2}

        distribution = {}

        for name, idx in channels.items():

            channel = image[:, :, idx]

            distribution[name] = {
                "Mean": round(float(np.mean(channel)), 2),
                "Std": round(float(np.std(channel)), 2),
                "Min": int(np.min(channel)),
                "Max": int(np.max(channel)),
                "Median": round(float(np.median(channel)), 2),
            }

        return distribution

    # ---------------------------------------------------------
    # Color Temperature
    # ---------------------------------------------------------

    @staticmethod
    def color_temperature(image):
        """
        Estimate relative color temperature.

        Returns 'warm', 'cool', or 'neutral'.
        """

        b_mean = np.mean(image[:, :, 0])
        r_mean = np.mean(image[:, :, 2])

        ratio = r_mean / (b_mean + 1e-6)

        if ratio > 1.2:
            return "warm"
        elif ratio < 0.8:
            return "cool"
        else:
            return "neutral"

    # ---------------------------------------------------------
    # Saturation Analysis
    # ---------------------------------------------------------

    @staticmethod
    def saturation_level(image):
        """
        Compute average saturation.
        """

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]

        return {
            "Mean Saturation": round(float(np.mean(saturation)), 2),
            "Max Saturation": int(np.max(saturation)),
            "Min Saturation": int(np.min(saturation)),
        }

    # ---------------------------------------------------------
    # Histogram Comparison
    # ---------------------------------------------------------

    @staticmethod
    def compare_histograms(image1, image2, method="correlation"):
        """
        Compare color histograms of two images.

        Parameters
        ----------
        method : str
            One of 'correlation', 'chi_square',
            'intersection', 'bhattacharyya'.

        Returns
        -------
        float
            Similarity score.
        """

        methods = {
            "correlation": cv2.HISTCMP_CORREL,
            "chi_square": cv2.HISTCMP_CHISQR,
            "intersection": cv2.HISTCMP_INTERSECT,
            "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
        }

        cv_method = methods.get(
            method, cv2.HISTCMP_CORREL
        )

        # Convert to HSV for better comparison
        hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist(
            [hsv1], [0, 1], None,
            [50, 60], [0, 180, 0, 256]
        )

        hist2 = cv2.calcHist(
            [hsv2], [0, 1], None,
            [50, 60], [0, 180, 0, 256]
        )

        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        score = cv2.compareHist(hist1, hist2, cv_method)

        return round(float(score), 4)

    # ---------------------------------------------------------
    # Full Color Analysis
    # ---------------------------------------------------------

    @staticmethod
    def full_analysis(image, k=5):
        """
        Run complete color analysis.

        Returns
        -------
        dict
            Complete color analysis results.
        """

        colors, percentages = ColorAnalysis.dominant_colors(
            image, k=k
        )

        palette = ColorAnalysis.create_palette(
            colors, percentages
        )

        distribution = ColorAnalysis.color_distribution(image)

        temperature = ColorAnalysis.color_temperature(image)

        saturation = ColorAnalysis.saturation_level(image)

        return {
            "dominant_colors": colors,
            "percentages": percentages,
            "palette": palette,
            "distribution": distribution,
            "temperature": temperature,
            "saturation": saturation,
        }
