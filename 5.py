#Explore a pretrained model (e.g., MobileNet) on a transfer learning task.

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

# Image preprocessing
train_data = ImageDataGenerator(rescale=1./255)

test_data = ImageDataGenerator(rescale=1./255)

# Load images
train = train_data.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test = test_data.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build transfer learning model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train,
    epochs=5,
    validation_data=test
)

# Evaluate model
loss, acc = model.evaluate(test)

print("Accuracy:", acc)