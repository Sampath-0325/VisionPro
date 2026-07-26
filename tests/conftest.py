"""
conftest.py
-----------
Shared pytest fixtures for VisionPro Studio tests.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def sample_bgr_image():
    """
    Create a simple 100x100 BGR test image
    with a colored gradient.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Blue gradient on left
    image[:, :50, 0] = np.linspace(0, 255, 100).reshape(100, 1).astype(np.uint8)

    # Green gradient on top
    image[:50, :, 1] = np.linspace(0, 255, 100).reshape(1, 100).astype(np.uint8)

    # Red block on bottom-right
    image[50:, 50:, 2] = 200

    return image


@pytest.fixture
def sample_gray_image():
    """
    Create a simple 100x100 grayscale test image.
    """
    image = np.zeros((100, 100), dtype=np.uint8)

    # Gradient
    for i in range(100):
        image[i, :] = int(i * 255 / 99)

    return image


@pytest.fixture
def sample_binary_image():
    """
    Create a simple binary test image with shapes.
    """
    image = np.zeros((100, 100), dtype=np.uint8)

    # Rectangle
    image[20:40, 30:70] = 255

    # Circle
    cv2.circle(image, (50, 70), 15, 255, -1)

    return image


@pytest.fixture
def sample_white_image():
    """
    Create a solid white 100x100 BGR image.
    """
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_black_image():
    """
    Create a solid black 100x100 BGR image.
    """
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_bgr_image():
    """
    Create a tiny 10x10 BGR image for fast tests.
    """
    return np.random.randint(
        0, 256, (10, 10, 3), dtype=np.uint8
    )


@pytest.fixture
def real_image_path():
    """
    Return path to a real test image if available.
    """
    test_images = [
        Path("images/Arches.jpeg"),
        Path("images/apple.jpg"),
        Path("images/ContourTest.jpg"),
    ]

    for path in test_images:
        if path.exists():
            return path

    return None


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Provide a temporary output directory.
    """
    output = tmp_path / "test_outputs"
    output.mkdir()
    return output
