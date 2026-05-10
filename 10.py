#Implement quantization and pruning techniques in a neural network to reduce its size and computational demands compare results with the baseline models


import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# ---------------- Baseline Model ----------------
baseline_model = Sequential([

    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Baseline Model")

baseline_model.fit(X_train, y_train, epochs=3)

baseline_loss, baseline_acc = baseline_model.evaluate(X_test, y_test)

print("Baseline Accuracy:", baseline_acc)


# ---------------- Pruning Model ----------------
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_model = prune_low_magnitude(baseline_model)

pruning_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Pruned Model")

pruning_model.fit(X_train, y_train, epochs=3)

prune_loss, prune_acc = pruning_model.evaluate(X_test, y_test)

print("Pruned Model Accuracy:", prune_acc)


# ---------------- Quantization Model ----------------
converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_model = converter.convert()

# Save quantized model
with open("quantized_model.tflite", "wb") as f:
    f.write(quantized_model)

print("\nQuantized model saved as quantized_model.tflite")


# ---------------- Comparison ----------------
print("\n----- Comparison -----")

print("Baseline Accuracy :", baseline_acc)
print("Pruned Accuracy   :", prune_acc)

print("\nQuantization reduces model size")
print("Pruning reduces number of parameters")