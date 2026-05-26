# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
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

from image_preprocessing import preprocess_image, IMAGE_SIZE

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


def build_compact_cnn(width, height, depth, classes):
    """
    Compact CNN optimized for 6-class road-sign recognition.
    Uses GlobalAveragePooling instead of a giant Flatten→Dense stack
    to drastically reduce parameters (~150k total vs ~25M before).
    BatchNormalization speeds up convergence and adds mild regularisation.
    """
    model = Sequential()
    inputShape = (height, width, depth)

    model.add(Input(inputShape))

    # Block 1: 32 filters
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2: 64 filters
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3: 128 filters
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block 4: 256 filters → GlobalAveragePooling collapses spatial dims
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.4))

    # Small fully-connected head
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

    
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
    
    
# preprocess_image already returns float32 data in [0, 1]
data = np.array(data, dtype="float32")
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=6)
testY = to_categorical(testY, num_classes=6)


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS  = 40
INIT_LR = 0.001
BS      = 32

# initialize the model
print("[INFO] compiling model...")
model = build_compact_cnn(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=1, classes=6)
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# print model summary
model.summary()

# checkpoint the best model
checkpoint_filepath = "..//network_model//model.best.keras"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = 'val_loss', verbose=1, save_best_only=True, mode='min')

# set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# callbacks
callbacks_list=[reduce_lr, checkpoint]

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), epochs=EPOCHS, callbacks=callbacks_list, verbose=1)
 
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