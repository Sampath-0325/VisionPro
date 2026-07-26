"""
filters.py
-----------
Image filtering operations for VisionPro Studio.
Includes blur, sharpen, emboss, artistic effects,
vignette, HDR, cartoon, and color quantization.
"""

import cv2
import numpy as np


class ImageFilters:

    # =========================================================
    # Blur Filters
    # =========================================================

    @staticmethod
    def gaussian_blur(image, kernel=(5, 5)):
        """
        Apply Gaussian Blur.
        """
        return cv2.GaussianBlur(image, kernel, 0)

    @staticmethod
    def median_blur(image, kernel_size=5):
        """
        Remove salt-and-pepper noise.
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def bilateral_filter(image,
                         diameter=9,
                         sigma_color=75,
                         sigma_space=75):
        """
        Smooth image while preserving edges.
        """
        return cv2.bilateralFilter(
            image,
            diameter,
            sigma_color,
            sigma_space
        )

    @staticmethod
    def box_blur(image, kernel=(5, 5)):
        """
        Average blur.
        """
        return cv2.blur(image, kernel)

    # =========================================================
    # Sharpening & Enhancement
    # =========================================================

    @staticmethod
    def sharpen(image):
        """
        Sharpen image using convolution.
        """

        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def unsharp_mask(image, sigma=1.0, strength=1.5):
        """
        Apply unsharp mask sharpening.

        Parameters
        ----------
        sigma : float
            Gaussian blur sigma.
        strength : float
            Sharpening strength.
        """

        blurred = cv2.GaussianBlur(
            image, (0, 0), sigma
        )

        sharpened = cv2.addWeighted(
            image, 1.0 + strength,
            blurred, -strength,
            0
        )

        return sharpened

    # =========================================================
    # Artistic Effects
    # =========================================================

    @staticmethod
    def emboss(image):
        """
        Emboss effect.
        """

        kernel = np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ])

        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def negative(image):
        """
        Negative image.
        """
        return cv2.bitwise_not(image)

    @staticmethod
    def sepia(image):
        """
        Apply Sepia effect.
        """

        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])

        sepia = cv2.transform(image, kernel)

        return np.clip(sepia, 0, 255).astype(np.uint8)

    @staticmethod
    def pencil_sketch(image):
        """
        Pencil sketch effect.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        inv = 255 - gray

        blur = cv2.GaussianBlur(inv, (21, 21), 0)

        inv_blur = 255 - blur

        sketch = cv2.divide(gray, inv_blur, scale=256)

        return sketch

    # =========================================================
    # Vignette Effect
    # =========================================================

    @staticmethod
    def vignette(image, strength=0.5):
        """
        Apply vignette (darkened edges) effect.

        Parameters
        ----------
        strength : float
            Vignette strength (0.0 to 1.0).
        """

        rows, cols = image.shape[:2]

        # Create Gaussian kernels for X and Y
        kernel_x = cv2.getGaussianKernel(cols, cols * 0.5)
        kernel_y = cv2.getGaussianKernel(rows, rows * 0.5)

        # Create 2D mask
        mask = kernel_y * kernel_x.T

        # Normalize to 0-1
        mask = mask / mask.max()

        # Blend with strength
        mask = (1 - strength) + strength * mask

        # Apply to each channel
        if len(image.shape) == 3:

            result = np.zeros_like(image, dtype=np.float64)

            for i in range(image.shape[2]):
                result[:, :, i] = image[:, :, i] * mask

        else:
            result = image * mask

        return np.clip(result, 0, 255).astype(np.uint8)

    # =========================================================
    # HDR Tone Mapping
    # =========================================================

    @staticmethod
    def hdr_effect(image, sigma_s=12, sigma_r=0.15):
        """
        Simulate HDR effect using edge-preserving filter.

        Parameters
        ----------
        sigma_s : float
            Spatial sigma.
        sigma_r : float
            Range sigma.
        """

        # Edge-preserving filter for HDR-like detail
        filtered = cv2.edgePreservingFilter(
            image,
            flags=1,
            sigma_s=sigma_s,
            sigma_r=sigma_r
        )

        # Enhance detail
        detail = cv2.detailEnhance(
            filtered,
            sigma_s=sigma_s,
            sigma_r=sigma_r
        )

        return detail

    # =========================================================
    # Cartoon Effect
    # =========================================================

    @staticmethod
    def cartoon(image, num_downsamples=2, num_bilateral=7):
        """
        Apply cartoon effect.

        Uses bilateral filtering for smoothing and
        adaptive thresholding for edges.
        """

        # Downsample for speed
        color = image.copy()

        for _ in range(num_downsamples):
            color = cv2.pyrDown(color)

        # Apply bilateral filter multiple times
        for _ in range(num_bilateral):
            color = cv2.bilateralFilter(
                color, d=9,
                sigmaColor=9,
                sigmaSpace=7
            )

        # Upsample back
        for _ in range(num_downsamples):
            color = cv2.pyrUp(color)

        # Ensure same size as original
        color = cv2.resize(
            color,
            (image.shape[1], image.shape[0])
        )

        # Convert to grayscale and get edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.medianBlur(gray, 7)

        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )

        # Combine color and edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cartoon = cv2.bitwise_and(color, edges_colored)

        return cartoon

    # =========================================================
    # Oil Painting Effect
    # =========================================================

    @staticmethod
    def oil_painting(image, size=7, dyn_ratio=1):
        """
        Apply oil painting effect.

        Parameters
        ----------
        size : int
            Neighborhood size.
        dyn_ratio : int
            Dynamic ratio for quantization.
        """

        # Use stylization as a fallback since
        # xphoto may not be available
        try:
            result = cv2.xphoto.oilPainting(
                image, size, dyn_ratio
            )
        except AttributeError:
            # Fallback: stylization filter
            result = cv2.stylization(
                image,
                sigma_s=60,
                sigma_r=0.6
            )

        return result

    # =========================================================
    # Color Quantization
    # =========================================================

    @staticmethod
    def color_quantize(image, k=8):
        """
        Reduce number of colors using K-Means.

        Parameters
        ----------
        k : int
            Number of colors to reduce to.
        """

        # Reshape to pixel list
        data = image.reshape(-1, 3).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20, 0.5
        )

        _, labels, centers = cv2.kmeans(
            data, k, None,
            criteria, 10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Map each pixel to its cluster center
        centers = np.uint8(centers)

        quantized = centers[labels.flatten()]

        return quantized.reshape(image.shape)

    # =========================================================
    # Warm / Cool Tone
    # =========================================================

    @staticmethod
    def warm_tone(image, intensity=20):
        """
        Apply warm color tone.
        """

        result = image.copy().astype(np.int16)

        result[:, :, 2] = np.clip(
            result[:, :, 2] + intensity, 0, 255
        )  # Red

        result[:, :, 0] = np.clip(
            result[:, :, 0] - intensity // 2, 0, 255
        )  # Blue

        return result.astype(np.uint8)

    @staticmethod
    def cool_tone(image, intensity=20):
        """
        Apply cool color tone.
        """

        result = image.copy().astype(np.int16)

        result[:, :, 0] = np.clip(
            result[:, :, 0] + intensity, 0, 255
        )  # Blue

        result[:, :, 2] = np.clip(
            result[:, :, 2] - intensity // 2, 0, 255
        )  # Red

        return result.astype(np.uint8)