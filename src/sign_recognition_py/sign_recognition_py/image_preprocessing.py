"""
Shared image preprocessing for road-sign recognition.

This module provides the canonical ``preprocess_image`` function used both
during CNN training and at inference time.  Keeping a single implementation
ensures that every node and script applies the *exact* same pipeline.
"""

import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Target image size used during training (width == height)
IMAGE_SIZE = 128


def preprocess_image(image):
    """
    Preprocess an image from the simulated gray environment.

    Pipeline
    --------
    1. Crop the upper-right corner square (side = half the image height).
       All road signs are located in that region.
    2. Convert to grayscale (single-channel black & white).
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to enhance local contrast in the predominantly gray scene.
    4. Resize to the fixed network input size (``IMAGE_SIZE``).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(IMAGE_SIZE, IMAGE_SIZE, 1)`` with pixel
        values in [0, 1].
    """
    h, w = image.shape[:2]
    square_side = h // 2

    # Crop upper-right corner: square of size (square_side x square_side)
    x_start = w - square_side
    y_start = 0
    cropped = image[y_start:y_start + square_side, x_start:x_start + square_side]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # CLAHE directly on the single-channel grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Resize to the fixed network input size
    resized = cv2.resize(equalized, (IMAGE_SIZE, IMAGE_SIZE))

    # img_to_array on a 2D grayscale image yields (H, W, 1)
    return img_to_array(resized) / 255.0
