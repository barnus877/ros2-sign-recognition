# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import __version__ as keras_version
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from numpy.random import seed

#from sign_recognition_py.image_preprocessing import preprocess_image, IMAGE_SIZE

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

    # Convert to RGB for MobileNetV2 preprocessing
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # img_to_array yields (H, W, 3)
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


def build_transfer_model(width, height, depth, classes):
    """
    MobileNetV2 transfer model with a compact classification head.
    Keeps file size low while boosting accuracy with pre-trained features.
    """
    inputShape = (height, width, depth)

    base_model = MobileNetV2(
        input_shape=inputShape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = Input(shape=inputShape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

    
dataset = '..//training_images'
# initialize the data and labels
print("[INFO] loading images and labels...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it (crop upper-right corner + CLAHE), and store it
    image = cv2.imread(imagePath)
    if image is None:
        print("[WARN] Could not read %s — skipping" % imagePath)
        continue
    image = preprocess_image(image)
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    print("Image: %s, Label: %s" % (imagePath, label))
    if label == 'limit_5':
        label = 0
    elif label == 'limit_40':
        label = 1
    elif label == 'limit_no':
        label = 2
    elif label == 'lived_place':
        label = 3
    elif label == 'no_sign':
        label = 4
    else:
        label = 5
    labels.append(label)
    
    
# preprocess_image returns float32 in [0, 255]
data = np.array(data, dtype="float32")
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY_int, testY_int) = train_test_split(
    data,
    labels,
    test_size=0.25,
    stratify=labels,
    random_state=42,
)

# convert the labels from integers to vectors
trainY = to_categorical(trainY_int, num_classes=6)
testY = to_categorical(testY_int, num_classes=6)

class_counts = np.bincount(trainY_int, minlength=6)
class_weight = {
    i: float(class_counts.max() / count) if count > 0 else 1.0
    for i, count in enumerate(class_counts)
}


# initialize the number of epochs to train for, initial learning rate,
# and batch size
WARMUP_EPOCHS = 12
FINE_TUNE_EPOCHS = 50
INIT_LR = 0.001
FINE_TUNE_LR = 0.0001
BS = 32

# initialize the model
print("[INFO] compiling model...")
model, base_model = build_transfer_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=3, classes=6)
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# print model summary
model.summary()

# checkpoint the best model
checkpoint_filepath = "..//network_model//model.best.keras"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = 'val_loss', verbose=1, save_best_only=True, mode='min')

# set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.5, min_lr=1e-6)

# stop training when the validation loss hasn't improved
early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True)

# callbacks
callbacks_list=[reduce_lr, checkpoint, early_stopping]

# data augmentation
aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode="nearest"
)

# train the network
print("[INFO] training network...")
train_gen = aug.flow(trainX, trainY, batch_size=BS)
val_data = (preprocess_input(testX.copy()), testY)

warmup_history = model.fit(
    train_gen,
    validation_data=val_data,
    epochs=WARMUP_EPOCHS,
    callbacks=[reduce_lr, checkpoint],
    class_weight=class_weight,
    verbose=1,
)

# Fine-tune the top layers of the backbone
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    metrics=["accuracy"],
)

history = model.fit(
    train_gen,
    validation_data=val_data,
    epochs=WARMUP_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(warmup_history.history["loss"]),
    callbacks=callbacks_list,
    class_weight=class_weight,
    verbose=1,
)
 
# save the model to disk
print("[INFO] serializing network...")
model.save("..//network_model//model.keras")

plt.xlabel('Epoch Number')
plt.ylabel("Loss / Accuracy Magnitude")
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['accuracy'], label="acc")
plt.plot(history.history['val_loss'], label="val_loss")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.legend()
plt.savefig('..//network_model//model_training')
plt.show()