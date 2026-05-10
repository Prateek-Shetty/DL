#Create a denoising autoencoder to remove noise from images.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Load MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Add noise
noise_factor = 0.5

X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(size=X_test.shape)

# Keep values between 0 and 1
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Build Autoencoder
model = Sequential([

    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    MaxPooling2D((2,2), padding='same'),

    Conv2D(32, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),

    Conv2D(1, (3,3), activation='sigmoid', padding='same')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train model
model.fit(
    X_train_noisy,
    X_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test_noisy, X_test)
)

# Predict denoised images
decoded_images = model.predict(X_test_noisy)

# Display results
plt.figure(figsize=(10,4))

for i in range(5):

    # Noisy image
    plt.subplot(2,5,i+1)
    plt.imshow(X_test_noisy[i].reshape(28,28), cmap='gray')
    plt.axis('off')

    # Denoised image
    plt.subplot(2,5,i+6)
    plt.imshow(decoded_images[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.show()