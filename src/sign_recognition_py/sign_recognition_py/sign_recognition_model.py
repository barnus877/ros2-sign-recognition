import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONSTANTS & CONFIGURATION ---
# 6 Classes: speed_5, speed_40, habited_area, stop, speed_lift_off, no_sign
NUM_CLASSES = 6
INPUT_SHAPE = (64, 64, 3) # Small resolution for high-speed inference
CLASS_NAMES = ['habited_area', 'no_sign', 'speed_40', 'speed_5', 'speed_lift_off', 'stop']

# --- PREPROCESSING FOR REAL-TIME INFERENCE ---
def preprocess_for_inference(image_array, target_size=(64, 64)):
    """
    Crops the top-right quarter of the image, resizes, and normalizes.
    Optimized for fast real-time execution in the ROS2 loop.
    """
    h, w = image_array.shape[:2]
    
    # 1. Crop top-right quarter (h//2, w//2)
    # Since signs are guaranteed to be here, we drop 75% of useless data (black line, floor)
    cropped_img = image_array[0:h//2, w//2:w]
    
    # 2. Resize to network input shape
    resized_img = cv2.resize(cropped_img, target_size)
    
    # 3. Normalize to [0, 1] range and expand dims for batch size of 1
    normalized_img = resized_img.astype('float32') / 255.0
    return np.expand_dims(normalized_img, axis=0)

# --- MODEL ARCHITECTURE ---
def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Builds a lightweight CNN.
    Due to simulation simplicity and cropping, a deep network (like ResNet) is overkill 
    and would violate the latency constraints. A 3-block CNN is sufficient.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), # Prevent overfitting on mostly gray background
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- TRAINING PIPELINE (Example) ---
def train_model(dataset_dir):
    """
    Handles training dataset loading and augmentation.
    Note: Ensure your dataset images are pre-cropped or use tf.image.crop_to_bounding_box
    in a map function to match the inference pipeline.
    """
    # Create dataset pipeline
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(480, 640), # Original size before cropping
        batch_size=32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(480, 640),
        batch_size=32
    )

    # Dataset map function to apply the exact same cropping as inference
    def tf_preprocess(images, labels):
        # Crop top right: tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        # Assuming 640x480 input -> crop top right 320x240
        cropped = tf.image.crop_to_bounding_box(images, 0, 320, 240, 320)
        resized = tf.image.resize(cropped, [64, 64])
        normalized = resized / 255.0
        return normalized, labels

    train_ds = raw_ds.map(tf_preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(tf_preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build and train
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=15)
    
    model.save('gazebo_sign_classifier.h5')
    return model

# --- USAGE EXAMPLE FOR ROS2 CALLBACK ---
def inference_callback(image_from_camera, model):
    """
    Example of how this integrates into your main loop.
    """
    # 1. Preprocess
    tensor = preprocess_for_inference(image_from_camera)
    
    # 2. Predict
    predictions = model.predict(tensor, verbose=0)
    
    # 3. Extract highest confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    return CLASS_NAMES[class_idx], confidence

if __name__ == "__main__":
    # To train: 
    # train_model("path/to/dataset_folder")
    pass