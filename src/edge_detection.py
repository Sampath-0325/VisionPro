"""
edge_detection.py
-----------------
Edge detection algorithms for VisionPro Studio.
Includes Sobel, Laplacian, Canny, Prewitt, Roberts,
Scharr, and edge overlay.
"""

import cv2
import numpy as np


class EdgeDetector:

    # ---------------------------------------------------------
    # Sobel
    # ---------------------------------------------------------

    @staticmethod
    def sobel(image, ksize=3):
        """
        Sobel edge detection.

        Returns
        -------
        tuple
            (sobel_x, sobel_y, combined)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        combined = cv2.addWeighted(
            sobel_x,
            0.5,
            sobel_y,
            0.5,
            0
        )

        return sobel_x, sobel_y, combined

    # ---------------------------------------------------------
    # Laplacian
    # ---------------------------------------------------------

    @staticmethod
    def laplacian(image):
        """
        Laplacian edge detection.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F)

        return cv2.convertScaleAbs(lap)

    # ---------------------------------------------------------
    # Canny
    # ---------------------------------------------------------

    @staticmethod
    def canny(
        image,
        threshold1=100,
        threshold2=200
    ):
        """
        Canny edge detection.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.Canny(
            gray,
            threshold1,
            threshold2
        )

    # ---------------------------------------------------------
    # Prewitt
    # ---------------------------------------------------------

    @staticmethod
    def prewitt(image):
        """
        Prewitt edge detection.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        kernel_y = np.array([
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1]
        ])

        x = cv2.filter2D(gray, -1, kernel_x)
        y = cv2.filter2D(gray, -1, kernel_y)

        return cv2.addWeighted(x, 0.5, y, 0.5, 0)

    # ---------------------------------------------------------
    # Roberts
    # ---------------------------------------------------------

    @staticmethod
    def roberts(image):
        """
        Roberts cross-gradient edge detection.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_x = np.array([
            [1,  0],
            [0, -1]
        ])

        kernel_y = np.array([
            [ 0, 1],
            [-1, 0]
        ])

        x = cv2.filter2D(gray, -1, kernel_x)
        y = cv2.filter2D(gray, -1, kernel_y)

        return cv2.addWeighted(x, 0.5, y, 0.5, 0)

    # ---------------------------------------------------------
    # Scharr
    # ---------------------------------------------------------

    @staticmethod
    def scharr(image):
        """
        Scharr edge detection.

        More accurate than Sobel for 3x3 kernels.

        Returns
        -------
        tuple
            (scharr_x, scharr_y, combined)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        scharr_x = cv2.convertScaleAbs(scharr_x)
        scharr_y = cv2.convertScaleAbs(scharr_y)

        combined = cv2.addWeighted(
            scharr_x, 0.5,
            scharr_y, 0.5,
            0
        )

        return scharr_x, scharr_y, combined

    # ---------------------------------------------------------
    # Edge Overlay
    # ---------------------------------------------------------

    @staticmethod
    def edge_overlay(
        image,
        edge_color=(0, 255, 0),
        threshold1=100,
        threshold2=200,
        thickness=1
    ):
        """
        Draw Canny edges overlaid on the original image.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image.
        edge_color : tuple
            BGR color for edges.
        threshold1 : int
            Canny lower threshold.
        threshold2 : int
            Canny upper threshold.
        thickness : int
            Edge line thickness.

        Returns
        -------
        numpy.ndarray
            Original image with edges drawn on top.
        """

        # Get edges
        edges = EdgeDetector.canny(
            image, threshold1, threshold2
        )

        # Create colored overlay
        output = image.copy()

        # Create edge mask
        if thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (thickness, thickness)
            )
            edges = cv2.dilate(edges, kernel)

        # Apply edge color
        output[edges > 0] = edge_color

        return output

    # ---------------------------------------------------------
    # Multi-scale Edge Detection
    # ---------------------------------------------------------

    @staticmethod
    def multi_scale_canny(
        image,
        scales=None
    ):
        """
        Canny edge detection at multiple scales.

        Parameters
        ----------
        scales : list[tuple] or None
            List of (threshold1, threshold2) pairs.
            Defaults to three sensitivity levels.

        Returns
        -------
        dict
            {"low": edges, "medium": edges, "high": edges}
        """

        if scales is None:
            scales = [
                (50, 100),    # high sensitivity
                (100, 200),   # medium
                (150, 300),   # low sensitivity
            ]

        labels = ["high_sensitivity", "medium", "low_sensitivity"]

        results = {}

        for label, (t1, t2) in zip(labels, scales):

            results[label] = EdgeDetector.canny(image, t1, t2)

        return results