import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Train baseline
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# Baseline accuracy
print("Baseline Accuracy:",
      model.evaluate(x_test, y_test, verbose=0)[1])

# Pruning
prune = tfmot.sparsity.keras.prune_low_magnitude
pruned_model = prune(model)

pruned_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

pruned_model.fit(x_train, y_train, epochs=1)

print("Pruned Accuracy:",
      pruned_model.evaluate(x_test, y_test, verbose=0)[1])

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_model = converter.convert()

print("Quantized Model Size:",
      len(quantized_model)/1024, "KB")