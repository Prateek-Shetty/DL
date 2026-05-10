#Experiment with different optimizers (e.g., Adam vs. RMSProp) and compare their impact on accuracy and convergence.


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Function to create model
def create_model():

    model = models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model


# -------- Adam Optimizer --------
adam_model = create_model()

adam_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with Adam Optimizer")

adam_history = adam_model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)


# -------- RMSProp Optimizer --------
rms_model = create_model()

rms_model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with RMSProp Optimizer")

rms_history = rms_model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)


# Final Accuracy Comparison
print("\nAdam Final Accuracy:",
      adam_history.history['val_accuracy'][-1])

print("RMSProp Final Accuracy:",
      rms_history.history['val_accuracy'][-1])