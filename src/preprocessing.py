"""
preprocessing.py
----------------
Image preprocessing and enhancement operations.
Includes grayscale, histogram equalization, brightness,
contrast, gamma, resize, rotate, flip, color conversions,
and noise generation.
"""

import cv2
import numpy as np


class Preprocessor:

    # =========================================================
    # Color Conversions
    # =========================================================

    @staticmethod
    def grayscale(image):
        """
        Convert BGR image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_hsv(image):
        """
        Convert BGR image to HSV.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def to_lab(image):
        """
        Convert BGR image to LAB color space.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def to_rgb(image):
        """
        Convert BGR to RGB (for display libraries).
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =========================================================
    # Histogram Enhancement
    # =========================================================

    @staticmethod
    def histogram_equalization(gray_image):
        """
        Improve image contrast via histogram equalization.
        """
        return cv2.equalizeHist(gray_image)

    @staticmethod
    def clahe(gray_image,
              clip_limit=2.0,
              tile_grid_size=(8, 8)):
        """
        Contrast Limited Adaptive Histogram Equalization.
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        return clahe.apply(gray_image)

    # =========================================================
    # Brightness & Contrast
    # =========================================================

    @staticmethod
    def adjust_brightness(image, beta=30):
        """
        Increase/decrease brightness.

        Parameters
        ----------
        beta : int
            Brightness offset (-255 to 255).
        """
        return cv2.convertScaleAbs(
            image,
            alpha=1,
            beta=beta
        )

    @staticmethod
    def adjust_contrast(image, alpha=1.3):
        """
        Increase/decrease contrast.

        Parameters
        ----------
        alpha : float
            Contrast multiplier (0.0 to 3.0).
        """
        return cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=0
        )

    @staticmethod
    def gamma_correction(image, gamma=1.2):
        """
        Apply gamma correction.

        Parameters
        ----------
        gamma : float
            Gamma value. >1 darkens, <1 brightens.
        """

        inv_gamma = 1.0 / gamma

        table = np.array(
            [
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(256)
            ]
        ).astype("uint8")

        return cv2.LUT(image, table)

    # =========================================================
    # Resize Operations
    # =========================================================

    @staticmethod
    def resize(image, width=None, height=None):
        """
        Resize image maintaining aspect ratio.

        Parameters
        ----------
        width : int or None
            Target width.
        height : int or None
            Target height.
        """

        h, w = image.shape[:2]

        if width is None and height is None:
            return image

        if width is not None and height is not None:
            return cv2.resize(image, (width, height))

        if width is not None:
            ratio = width / w
            dimension = (width, int(h * ratio))
        else:
            ratio = height / h
            dimension = (int(w * ratio), height)

        return cv2.resize(image, dimension)

    @staticmethod
    def resize_exact(image, width, height):
        """
        Resize image to exact dimensions (may distort).
        """
        return cv2.resize(image, (width, height))

    @staticmethod
    def crop(image, x, y, w, h):
        """
        Crop a rectangular region from the image.

        Parameters
        ----------
        x, y : int
            Top-left corner coordinates.
        w, h : int
            Width and height of crop region.
        """
        return image[y:y + h, x:x + w].copy()

    @staticmethod
    def center_crop(image, crop_width, crop_height):
        """
        Crop from the center of the image.
        """

        h, w = image.shape[:2]

        x = max(0, (w - crop_width) // 2)
        y = max(0, (h - crop_height) // 2)

        return image[y:y + crop_height, x:x + crop_width].copy()

    # =========================================================
    # Rotation & Flip
    # =========================================================

    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        """
        Rotate image by a given angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees.
        center : tuple or None
            Center of rotation. Image center if None.
        scale : float
            Scale factor.
        """

        h, w = image.shape[:2]

        if center is None:
            center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center, angle, scale
        )

        return cv2.warpAffine(image, matrix, (w, h))

    @staticmethod
    def rotate_90(image):
        """Rotate 90 degrees clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def rotate_180(image):
        """Rotate 180 degrees."""
        return cv2.rotate(image, cv2.ROTATE_180)

    @staticmethod
    def rotate_270(image):
        """Rotate 270 degrees (90 counter-clockwise)."""
        return cv2.rotate(
            image, cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    @staticmethod
    def flip_horizontal(image):
        """Flip image horizontally."""
        return cv2.flip(image, 1)

    @staticmethod
    def flip_vertical(image):
        """Flip image vertically."""
        return cv2.flip(image, 0)

    @staticmethod
    def flip_both(image):
        """Flip both horizontally and vertically."""
        return cv2.flip(image, -1)

    # =========================================================
    # Noise Generation (for testing denoising)
    # =========================================================

    @staticmethod
    def add_gaussian_noise(image, mean=0, sigma=25):
        """
        Add Gaussian noise to image.

        Parameters
        ----------
        mean : float
            Mean of the noise.
        sigma : float
            Standard deviation of the noise.
        """

        noise = np.random.normal(
            mean, sigma, image.shape
        ).astype(np.float64)

        noisy = image.astype(np.float64) + noise

        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise(image, amount=0.02):
        """
        Add salt-and-pepper noise to image.

        Parameters
        ----------
        amount : float
            Proportion of pixels affected (0.0 to 1.0).
        """

        output = image.copy()

        # Salt (white pixels)
        num_salt = int(amount * image.size / 2)

        coords = [
            np.random.randint(0, i - 1, num_salt)
            for i in image.shape[:2]
        ]

        output[coords[0], coords[1]] = 255

        # Pepper (black pixels)
        num_pepper = int(amount * image.size / 2)

        coords = [
            np.random.randint(0, i - 1, num_pepper)
            for i in image.shape[:2]
        ]

        output[coords[0], coords[1]] = 0

        return output

    @staticmethod
    def add_speckle_noise(image):
        """
        Add speckle (multiplicative) noise.
        """

        noise = np.random.randn(*image.shape).astype(np.float64)

        noisy = image.astype(np.float64) + image.astype(np.float64) * noise * 0.1

        return np.clip(noisy, 0, 255).astype(np.uint8)