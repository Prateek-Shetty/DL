import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU

# Load data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5
x_train = x_train.reshape(-1,28,28,1)

# Generator
generator = Sequential([
    Dense(256, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='tanh'),
    Reshape((28,28,1))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(256),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training
for epoch in range(1000):

    # Real images
    idx = np.random.randint(0, x_train.shape[0], 64)
    real = x_train[idx]

    # Fake images
    noise = np.random.normal(0,1,(64,100))
    fake = generator.predict(noise, verbose=0)

    # Train discriminator
    discriminator.train_on_batch(real, np.ones((64,1)))
    discriminator.train_on_batch(fake, np.zeros((64,1)))

    # Train generator
    noise = np.random.normal(0,1,(64,100))
    gan.train_on_batch(noise, np.ones((64,1)))

# Generate image
noise = np.random.normal(0,1,(1,100))
img = generator.predict(noise, verbose=0)

plt.imshow(img[0,:,:,0], cmap='gray')
plt.axis('off')
plt.show()