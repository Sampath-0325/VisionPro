"""
==============================================================
VisionPro Studio
Image Metrics & Analysis Module
==============================================================

This module calculates image statistics, quality metrics,
histogram information, PSNR, SSIM, and color analysis.

Author : VisionPro Studio Team
"""

import cv2
import numpy as np


class ImageMetrics:

    # =========================================================
    # Resolution & Shape
    # =========================================================

    @staticmethod
    def resolution(image):
        """
        Return image resolution.
        """
        height, width = image.shape[:2]

        return {
            "Width": width,
            "Height": height,
            "Resolution": f"{width} x {height}"
        }

    @staticmethod
    def channels(image):
        """
        Return number of channels.
        """
        if len(image.shape) == 2:
            return 1

        return image.shape[2]

    # =========================================================
    # Intensity Statistics
    # =========================================================

    @staticmethod
    def brightness(image):
        """
        Average brightness.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(float(np.mean(gray)), 2)

    @staticmethod
    def contrast(image):
        """
        Standard deviation of intensity (contrast).
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(float(np.std(gray)), 2)

    @staticmethod
    def sharpness(image):
        """
        Variance of Laplacian (sharpness measure).
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var(),
            2
        )

    @staticmethod
    def entropy(image):
        """
        Shannon entropy.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        histogram = histogram / histogram.sum()

        histogram = histogram[
            histogram > 0
        ]

        entropy = -np.sum(
            histogram * np.log2(histogram)
        )

        return round(float(entropy), 3)

    @staticmethod
    def mean(image):
        """
        Mean intensity.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(float(np.mean(gray)), 2)

    @staticmethod
    def median(image):
        """
        Median intensity.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(float(np.median(gray)), 2)

    @staticmethod
    def variance(image):
        """
        Variance.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return round(float(np.var(gray)), 2)

    @staticmethod
    def minimum(image):
        """
        Minimum pixel value.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return int(np.min(gray))

    @staticmethod
    def maximum(image):
        """
        Maximum pixel value.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return int(np.max(gray))

    # =========================================================
    # Histograms
    # =========================================================

    @staticmethod
    def histogram(image):
        """
        Calculate grayscale histogram.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        return histogram

    @staticmethod
    def rgb_histogram(image):
        """
        RGB Histograms.
        """

        histograms = {}

        colors = (
            "blue",
            "green",
            "red"
        )

        for i, color in enumerate(colors):

            histograms[color] = cv2.calcHist(
                [image],
                [i],
                None,
                [256],
                [0, 256]
            )

        return histograms

    # =========================================================
    # Image Quality Metrics
    # =========================================================

    @staticmethod
    def psnr(original, processed):
        """
        Peak Signal-to-Noise Ratio (PSNR).

        Parameters
        ----------
        original : numpy.ndarray
            Reference image.
        processed : numpy.ndarray
            Processed/distorted image.

        Returns
        -------
        float
            PSNR value in dB. Higher is better.
            Returns float('inf') if images are identical.
        """

        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(
                processed,
                (original.shape[1], original.shape[0])
            )

        # Convert both to same type for comparison
        if len(original.shape) == 3 and len(processed.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif len(original.shape) == 2 and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        mse = np.mean(
            (original.astype(np.float64) - processed.astype(np.float64)) ** 2
        )

        if mse == 0:
            return float("inf")

        max_pixel = 255.0

        psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))

        return round(float(psnr_val), 2)

    @staticmethod
    def ssim(original, processed):
        """
        Structural Similarity Index (SSIM).

        A simplified implementation without external dependencies.

        Parameters
        ----------
        original : numpy.ndarray
            Reference image.
        processed : numpy.ndarray
            Processed/distorted image.

        Returns
        -------
        float
            SSIM value between -1 and 1. Higher is better.
        """

        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(
                processed,
                (original.shape[1], original.shape[0])
            )

        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        original = original.astype(np.float64)
        processed = processed.astype(np.float64)

        # Constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        # Mean
        mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(processed, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variance
        sigma1_sq = cv2.GaussianBlur(
            original ** 2, (11, 11), 1.5
        ) - mu1_sq

        sigma2_sq = cv2.GaussianBlur(
            processed ** 2, (11, 11), 1.5
        ) - mu2_sq

        sigma12 = cv2.GaussianBlur(
            original * processed, (11, 11), 1.5
        ) - mu1_mu2

        # SSIM formula
        numerator = (
            (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        )

        denominator = (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        ssim_map = numerator / denominator

        return round(float(np.mean(ssim_map)), 4)

    # =========================================================
    # Dominant Color Detection
    # =========================================================

    @staticmethod
    def dominant_colors(image, k=5):
        """
        Find dominant colors using K-Means clustering.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image.
        k : int
            Number of dominant colors.

        Returns
        -------
        list[dict]
            List of dominant colors with BGR values and
            percentage.
        """

        pixels = image.reshape(-1, 3).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.2
        )

        _, labels, centers = cv2.kmeans(
            pixels, k, None,
            criteria, 10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        _, counts = np.unique(labels, return_counts=True)

        percentages = counts / counts.sum() * 100

        sort_idx = np.argsort(-percentages)

        results = []

        for idx in sort_idx:

            bgr = centers[idx].astype(int).tolist()

            rgb = [bgr[2], bgr[1], bgr[0]]

            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

            results.append({
                "BGR": bgr,
                "RGB": rgb,
                "Hex": hex_color,
                "Percentage": round(float(percentages[idx]), 2)
            })

        return results

    # =========================================================
    # Color Distribution Analysis
    # =========================================================

    @staticmethod
    def color_distribution(image):
        """
        Analyze color distribution across BGR channels.

        Returns
        -------
        dict
            Statistics for each color channel.
        """

        channels = {"Blue": 0, "Green": 1, "Red": 2}

        distribution = {}

        for name, idx in channels.items():

            channel = image[:, :, idx].astype(np.float64)

            distribution[name] = {
                "Mean": round(float(np.mean(channel)), 2),
                "Std": round(float(np.std(channel)), 2),
                "Min": int(np.min(channel)),
                "Max": int(np.max(channel)),
            }

        return distribution

    # =========================================================
    # Complete Statistics
    # =========================================================

    @staticmethod
    def image_statistics(image):
        """
        Complete image analysis.
        """

        stats = {

            "Resolution":
                ImageMetrics.resolution(image),

            "Channels":
                ImageMetrics.channels(image),

            "Brightness":
                ImageMetrics.brightness(image),

            "Contrast":
                ImageMetrics.contrast(image),

            "Sharpness":
                ImageMetrics.sharpness(image),

            "Entropy":
                ImageMetrics.entropy(image),

            "Mean":
                ImageMetrics.mean(image),

            "Median":
                ImageMetrics.median(image),

            "Variance":
                ImageMetrics.variance(image),

            "Minimum":
                ImageMetrics.minimum(image),

            "Maximum":
                ImageMetrics.maximum(image),

            "Color Distribution":
                ImageMetrics.color_distribution(image),

        }

        return stats