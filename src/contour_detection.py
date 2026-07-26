"""
==============================================================
VisionPro Studio
Contour Detection Module
==============================================================

Contour finding, drawing, analysis, and geometric
property extraction.

Author : VisionPro Studio Team
"""

import cv2
import numpy as np


class ContourDetector:
    """
    Contour detection and analysis operations.
    """

    # ---------------------------------------------------------
    # Find Contours
    # ---------------------------------------------------------

    @staticmethod
    def find_contours(
        image,
        threshold=127,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    ):
        """
        Find contours in an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image.
        threshold : int
            Binary threshold value.
        mode : int
            Contour retrieval mode.
        method : int
            Contour approximation method.

        Returns
        -------
        tuple
            (contours, hierarchy)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        contours, hierarchy = cv2.findContours(
            binary, mode, method
        )

        return contours, hierarchy

    # ---------------------------------------------------------
    # Draw Contours
    # ---------------------------------------------------------

    @staticmethod
    def draw_contours(
        image,
        contours,
        color=(0, 255, 0),
        thickness=2
    ):
        """
        Draw contours on a copy of the image.
        """

        output = image.copy()

        cv2.drawContours(
            output, contours, -1,
            color, thickness
        )

        return output

    # ---------------------------------------------------------
    # Contour Properties
    # ---------------------------------------------------------

    @staticmethod
    def contour_properties(contours):
        """
        Calculate properties for each contour.

        Returns
        -------
        list[dict]
            List of dictionaries with area, perimeter,
            bounding rect, centroid, circularity.
        """

        properties = []

        for i, contour in enumerate(contours):

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Bounding Rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Centroid via moments
            moments = cv2.moments(contour)

            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = 0, 0

            # Circularity
            circularity = 0.0
            if perimeter > 0:
                circularity = round(
                    4 * np.pi * area / (perimeter ** 2), 4
                )

            # Aspect ratio
            aspect_ratio = round(w / h, 3) if h > 0 else 0

            properties.append({
                "Index": i,
                "Area": round(area, 2),
                "Perimeter": round(perimeter, 2),
                "Bounding Box": (x, y, w, h),
                "Centroid": (cx, cy),
                "Circularity": circularity,
                "Aspect Ratio": aspect_ratio,
            })

        return properties

    # ---------------------------------------------------------
    # Bounding Rectangles
    # ---------------------------------------------------------

    @staticmethod
    def draw_bounding_boxes(
        image,
        contours,
        color=(255, 0, 0),
        thickness=2,
        min_area=100
    ):
        """
        Draw bounding rectangles around contours.
        """

        output = image.copy()

        for contour in contours:

            if cv2.contourArea(contour) < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                color,
                thickness
            )

        return output

    # ---------------------------------------------------------
    # Rotated Bounding Rectangles
    # ---------------------------------------------------------

    @staticmethod
    def draw_rotated_boxes(
        image,
        contours,
        color=(0, 0, 255),
        thickness=2,
        min_area=100
    ):
        """
        Draw minimum-area rotated rectangles.
        """

        output = image.copy()

        for contour in contours:

            if cv2.contourArea(contour) < min_area:
                continue

            rect = cv2.minAreaRect(contour)

            box = cv2.boxPoints(rect)

            box = np.intp(box)

            cv2.drawContours(
                output, [box], 0,
                color, thickness
            )

        return output

    # ---------------------------------------------------------
    # Convex Hull
    # ---------------------------------------------------------

    @staticmethod
    def convex_hull(
        image,
        contours,
        color=(255, 255, 0),
        thickness=2
    ):
        """
        Draw convex hulls around contours.
        """

        output = image.copy()

        hulls = [
            cv2.convexHull(c) for c in contours
        ]

        cv2.drawContours(
            output, hulls, -1,
            color, thickness
        )

        return output

    # ---------------------------------------------------------
    # Approximate Polygons
    # ---------------------------------------------------------

    @staticmethod
    def approximate_contours(
        image,
        contours,
        epsilon_factor=0.02,
        color=(0, 128, 255),
        thickness=2
    ):
        """
        Approximate contours with polygons.
        """

        output = image.copy()

        for contour in contours:

            epsilon = epsilon_factor * cv2.arcLength(
                contour, True
            )

            approx = cv2.approxPolyDP(
                contour, epsilon, True
            )

            cv2.drawContours(
                output, [approx], -1,
                color, thickness
            )

        return output

    # ---------------------------------------------------------
    # Minimum Enclosing Circle
    # ---------------------------------------------------------

    @staticmethod
    def min_enclosing_circles(
        image,
        contours,
        color=(128, 0, 255),
        thickness=2,
        min_area=100
    ):
        """
        Draw minimum enclosing circles around contours.
        """

        output = image.copy()

        for contour in contours:

            if cv2.contourArea(contour) < min_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            center = (int(x), int(y))

            radius = int(radius)

            cv2.circle(
                output, center, radius,
                color, thickness
            )

        return output

    # ---------------------------------------------------------
    # Full Analysis
    # ---------------------------------------------------------

    @staticmethod
    def full_analysis(image, threshold=127, min_area=100):
        """
        Run complete contour analysis and return all results.

        Returns
        -------
        dict
            Dictionary with all contour visualization results
            and properties.
        """

        contours, hierarchy = ContourDetector.find_contours(
            image, threshold
        )

        # Filter small contours
        filtered = [
            c for c in contours
            if cv2.contourArea(c) >= min_area
        ]

        results = {

            "contours_drawn":
                ContourDetector.draw_contours(
                    image, filtered
                ),

            "bounding_boxes":
                ContourDetector.draw_bounding_boxes(
                    image, filtered, min_area=0
                ),

            "rotated_boxes":
                ContourDetector.draw_rotated_boxes(
                    image, filtered, min_area=0
                ),

            "convex_hull":
                ContourDetector.convex_hull(
                    image, filtered
                ),

            "approx_polygons":
                ContourDetector.approximate_contours(
                    image, filtered
                ),

            "enclosing_circles":
                ContourDetector.min_enclosing_circles(
                    image, filtered, min_area=0
                ),
        }

        properties = ContourDetector.contour_properties(
            filtered
        )

        return results, properties
