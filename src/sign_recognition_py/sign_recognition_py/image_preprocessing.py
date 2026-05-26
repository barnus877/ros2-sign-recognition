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
    1. Crop the upper-right quadrant.
       All road signs are located in that region.
    2. Keep in color (BGR/RGB) instead of grayscale so that the neural
       network can distinguish red rings of signs from the gray environment.
    3. Resize to the fixed network input size (``IMAGE_SIZE``).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(IMAGE_SIZE, IMAGE_SIZE, 3)`` with pixel
        values in [0, 1].
    """
    h, w = image.shape[:2]

    # Crop upper-right quadrant
    # Signs are generally in the top-right quarter of the image
    y_start = 0
    y_end = h // 2
    x_start = w // 2
    x_end = w
    cropped = image[y_start:y_end, x_start:x_end]

    # Resize to the fixed network input size
    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

    # img_to_array yields (H, W, 3) for RGB/BGR
    return img_to_array(resized) / 255.0
