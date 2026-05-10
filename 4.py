#Fine-tune a pretrained model like ResNet50 or EfficientNet on a custom dataset.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import os

# Dataset path
train_dir = "dataset/train"
val_dir = "dataset/val"

# Image generators
train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)

val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)

# Load pretrained ResNet50
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

# Build model
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(len(os.listdir(train_dir)), activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("model.h5")