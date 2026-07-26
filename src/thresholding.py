"""
thresholding.py
----------------
Thresholding operations for VisionPro Studio.

This module provides different thresholding techniques used for
image segmentation and preprocessing.

Author: VisionPro Studio
"""

import cv2


class Thresholding:
    """
    Collection of thresholding algorithms.
    """

    @staticmethod
    def binary(image, threshold=127, max_value=255):
        """
        Binary Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            threshold,
            max_value,
            cv2.THRESH_BINARY
        )

        return binary

    @staticmethod
    def binary_inverse(image, threshold=127, max_value=255):
        """
        Binary Inverse Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            threshold,
            max_value,
            cv2.THRESH_BINARY_INV
        )

        return binary

    @staticmethod
    def truncate(image, threshold=127):
        """
        Truncate Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, truncate = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_TRUNC
        )

        return truncate

    @staticmethod
    def to_zero(image, threshold=127):
        """
        To-Zero Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, to_zero = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_TOZERO
        )

        return to_zero

    @staticmethod
    def adaptive_mean(
        image,
        block_size=11,
        c=2
    ):
        """
        Adaptive Mean Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )

    @staticmethod
    def adaptive_gaussian(
        image,
        block_size=11,
        c=2
    ):
        """
        Adaptive Gaussian Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )

    @staticmethod
    def otsu(image):
        """
        Otsu Thresholding
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return otsu

    @staticmethod
    def otsu_gaussian(image):
        """
        Gaussian Blur + Otsu Thresholding
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, otsu = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return otsu