# import the necessary packages
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import __version__ as keras_version
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.random import set_seed
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.keras.preprocessing.image import img_to_array

# Target image size used during training (width == height)
IMAGE_SIZE = 64
NUM_CLASSES = 6

# Class label mapping
CLASS_LABELS = {
    0: 'limit_5',
    1: 'limit_40',
    2: 'limit_no',
    3: 'lived_place',
    4: 'no_sign',
    5: 'stop',
}

# Set to True to continue training from the best previously saved model
CONTINUE_TRAINING = False

def preprocess_image(image):
    """
    Preprocess an image from the simulated gray environment.
    """
    h, w = image.shape[:2]

    # Crop upper-right quadrant
    cropped = image[0:h // 2, w // 2:w]

    # Resize to the fixed network input size
    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Return as 0-255 float32 array. Normalization will happen during augmentation/training.
    return img_to_array(rgb)


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Fix every random seed to make the training reproducible
seed(1)
set_seed(2)
random.seed(4)

print("[INFO] Version:")
print("Tensorflow version: %s" % tf.__version__)
keras_version = str(keras_version).encode('utf8')
print("Keras version: %s" % keras_version)


def build_model(input_shape, num_classes):
    """
    Restored Flatten for spatial awareness. 
    Removed BatchNormalization which was killing gradients.
    """
    model = Sequential([
        # Block 1
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classification Head
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4), # Balances overfitting without starving the network
        Dense(num_classes, activation='softmax')
    ])
    
    return model

    
dataset = '..//training_images'
print("[INFO] loading images and labels...")
data = []
labels = []
 
imagePaths = sorted(list(paths.list_images(dataset)))
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    if image is None:
        print("[WARN] Could not read %s — skipping" % imagePath)
        continue
    image = preprocess_image(image)
    data.append(image)
    
    label_str = imagePath.split(os.path.sep)[-2]
    
    if label_str == 'limit_5': label = 0
    elif label_str == 'limit_40': label = 1
    elif label_str == 'limit_no': label = 2
    elif label_str == 'lived_place': label = 3
    elif label_str == 'no_sign': label = 4
    else: label = 5
    labels.append(label)
    
# Data is float32 [0, 255] from preprocess_image
data = np.array(data)
labels = np.array(labels)
 
(trainX, testX, trainY_int, testY_int) = train_test_split(
    data, labels, test_size=0.25, stratify=labels, random_state=42
)

trainY = to_categorical(trainY_int, num_classes=NUM_CLASSES)
testY = to_categorical(testY_int, num_classes=NUM_CLASSES)

class_counts = np.bincount(trainY_int, minlength=NUM_CLASSES)
class_weight = {
    i: float(class_counts.max() / count) if count > 0 else 1.0
    for i, count in enumerate(class_counts)
}

print("[INFO] Class weights:", class_weight)

EPOCHS = 150
if CONTINUE_TRAINING:
    INIT_LR = 0.0001
else:
    INIT_LR = 0.001
BS = 32

print("[INFO] compiling model...")
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

if CONTINUE_TRAINING:
    checkpoint_filepath = "..//network_model//model.best.keras"
    if os.path.exists(checkpoint_filepath):
        print("[INFO] Loading previously saved best model for continued training...")
        model = load_model(checkpoint_filepath)
    else:
        print("[WARN] Best model checkpoint not found, building new model...")
        model = build_model(input_shape, NUM_CLASSES)
else:
    print("[INFO] Building new model...")
    model = build_model(input_shape, NUM_CLASSES)

opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
model.summary()

os.makedirs("..//network_model", exist_ok=True)
checkpoint_filepath = "..//network_model//model.best.keras"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=1e-6)

if CONTINUE_TRAINING:
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

callbacks_list = [reduce_lr, checkpoint, early_stopping]

# Use rescale here to normalize AFTER augmentation ops (like brightness) execute on [0, 255] data
aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode="nearest"
)

print("[INFO] training network...")
train_gen = aug.flow(trainX, trainY, batch_size=BS)

# Manually normalize test data
testX_normalized = testX / 255.0

history = model.fit(
    train_gen,
    validation_data=(testX_normalized, testY),
    epochs=EPOCHS,
    callbacks=callbacks_list,
    class_weight=class_weight,
    verbose=1,
)
 
print("[INFO] serializing network...")
model.save("..//network_model//model.keras")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('..//network_model//model_training.png')
plt.show()