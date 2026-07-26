"""
==========================================================
VisionPro Studio
Morphological Image Processing Module
==========================================================

This module contains various morphological operations used
for image enhancement and segmentation.

Author : VisionPro Studio Team
"""

import cv2
import numpy as np


class Morphology:
    """
    Morphological image processing operations.
    """

    @staticmethod
    def get_kernel(kernel_size=(5, 5)):
        """
        Create a rectangular structuring element.
        """
        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size
        )

    @staticmethod
    def erosion(image, kernel_size=(5, 5), iterations=1):
        """
        Perform erosion.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.erode(
            gray,
            kernel,
            iterations=iterations
        )

    @staticmethod
    def dilation(image, kernel_size=(5, 5), iterations=1):
        """
        Perform dilation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.dilate(
            gray,
            kernel,
            iterations=iterations
        )

    @staticmethod
    def opening(image, kernel_size=(5, 5)):
        """
        Remove small noise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel
        )

    @staticmethod
    def closing(image, kernel_size=(5, 5)):
        """
        Fill small holes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

    @staticmethod
    def gradient(image, kernel_size=(5, 5)):
        """
        Morphological Gradient.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            kernel
        )

    @staticmethod
    def top_hat(image, kernel_size=(9, 9)):
        """
        Top Hat transformation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            kernel
        )

    @staticmethod
    def black_hat(image, kernel_size=(9, 9)):
        """
        Black Hat transformation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = Morphology.get_kernel(kernel_size)

        return cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            kernel
        )

    @staticmethod
    def hit_or_miss(image):
        """
        Hit-or-Miss transformation.
        Works only on binary images.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.array([
            [0, 1, 0],
            [1, -1, 1],
            [0, 1, 0]
        ], dtype=np.int8)

        return cv2.morphologyEx(
            binary,
            cv2.MORPH_HITMISS,
            kernel
        )