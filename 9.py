#Implement a simple GAN to generate images from random noise (e.g., MNISTdigit generation).


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize images
X_train = X_train / 127.5 - 1.0
X_train = X_train.reshape(X_train.shape[0], 784)

# ---------------- Generator ----------------
generator = Sequential([

    Dense(256, input_dim=100),
    LeakyReLU(0.2),

    Dense(512),
    LeakyReLU(0.2),

    Dense(784, activation='tanh')
])

# ---------------- Discriminator ----------------
discriminator = Sequential([

    Dense(512, input_dim=784),
    LeakyReLU(0.2),

    Dense(256),
    LeakyReLU(0.2),

    Dense(1, activation='sigmoid')
])

# Compile discriminator
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- GAN Model ----------------
discriminator.trainable = False

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Training
epochs = 1000
batch_size = 64

for epoch in range(epochs):

    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(
        real_images,
        np.ones((batch_size, 1))
    )

    d_loss_fake = discriminator.train_on_batch(
        fake_images,
        np.zeros((batch_size, 1))
    )

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))

    g_loss = gan.train_on_batch(
        noise,
        np.ones((batch_size, 1))
    )

    # Print progress
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Generator Loss:", g_loss)

# Generate images
noise = np.random.normal(0, 1, (5, 100))
generated_images = generator.predict(noise)

# Display generated images
plt.figure(figsize=(10,2))

for i in range(5):

    plt.subplot(1,5,i+1)
    plt.imshow(generated_images[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.show()