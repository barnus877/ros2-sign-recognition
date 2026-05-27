import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import __version__ as keras_version
import tensorflow as tf
import zipfile
import json

import cv2
import numpy as np
import time


import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Target image size used during training (width == height)
IMAGE_SIZE = 64


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
        values in [0, 255].
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

    # Convert to RGB for network preprocessing
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # img_to_array yields (H, W, 3)
    return img_to_array(rgb)


# Class index → human-readable label (matching train_network.py mapping)
CLASS_LABELS = {
    0: 'limit_5',
    1: 'limit_40',
    2: 'limit_no',
    3: 'lived_place',
    4: 'no_sign',
    5: 'stop',
}




class SignRecogniser(Node):

    def __init__(self):
        super().__init__('sign_recogniser')

        # Declare a parameter for the evaluation interval (seconds)
        self.declare_parameter('eval_interval', 0.1)
        self.eval_interval = self.get_parameter('eval_interval').value
        self.get_logger().info(f'Evaluation interval set to {self.eval_interval:.3f} s')

        # --- TensorFlow session ---
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)

        # --- Locate and load the model ---
        pkg_share = get_package_share_directory('sign_recognition_py')
        model_path = pkg_share + '/network_model/model.best.keras'

        self.get_logger().info(f'TensorFlow version: {tf.__version__}')
        self.get_logger().info(f'Keras version: {keras_version}')
        self.get_logger().info(f'Loading model from: {model_path}')

        model_version = self._get_keras_version_from_keras_file(model_path)
        self.get_logger().info(f"Model's Keras version: {model_version}")

        keras_version_str = str(keras_version)
        if model_version != keras_version_str:
            self.get_logger().error(
                f'Keras version mismatch: runtime={keras_version_str}, '
                f'model={model_version}'
            )
            raise SystemExit(1)

        self.model = load_model(model_path, custom_objects=None, compile=True, safe_mode=True)
        self.model.summary()

        # --- CvBridge ---
        self.bridge = CvBridge()

        # --- Subscriber: same image stream as line_follower_cnn ---
        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed',
            self.image_callback,
            1,
        )

        # --- Latest frame storage ---
        self.latest_frame = None

        # --- Timing: throttle evaluation to eval_interval ---
        self.last_eval_time = 0.0

        # --- Processing timer: calls process() at the configured interval ---
        self.timer = self.create_timer(self.eval_interval, self.process)

        self.get_logger().info('Sign recogniser node started')

    def image_callback(self, msg):
        """Store the latest compressed image."""
        self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

    def process(self):
        """
        Timer callback — runs at eval_interval Hz.
        Preprocesses the latest frame with the same pipeline used during
        training and runs inference.  Prints the recognised sign label.
        """
        if self.latest_frame is None:
            return

        now = time.time()
        elapsed = now - self.last_eval_time
        self.last_eval_time = now

        # --- Preprocessing (identical to train_network.py) ---
        preprocessed = preprocess_image(self.latest_frame)

        # Add batch dimension: (64, 64, 3) → (1, 64, 64, 3)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        
        # Normalize to [0, 1] range to match training data normalization
        input_tensor = input_tensor / 255.0

        # --- Inference ---
        prediction = np.argmax(self.model(input_tensor, training=False), axis=1)[0]
        confidence = np.max(self.model(input_tensor, training=False), axis=1)[0]

        label = CLASS_LABELS.get(prediction, f'unknown({prediction})')

        self.get_logger().info(
            f'Sign: {label:<12s} | confidence: {confidence:.4f} | '
            f'inference took {elapsed*1000:.1f} ms'
        )

    @staticmethod
    def _get_keras_version_from_keras_file(path):
        """Extract Keras version stored inside a .keras archive."""
        with zipfile.ZipFile(path, 'r') as archive:
            if 'metadata.json' in archive.namelist():
                with archive.open('metadata.json') as f:
                    metadata = json.load(f)
                    return metadata.get('keras_version', 'Unknown')
            return 'Unknown'


def main(args=None):
    rclpy.init(args=args)
    node = SignRecogniser()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#the preprocessing was changed, but will be reverted.
# I made some changes in the preprocessing in the #file:train_network.py file. now the image is only converted into grayscale instead of binary format.