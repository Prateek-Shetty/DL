#Create a denoising autoencoder to remove noise from images.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train,_), (X_test,_) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
X_train, X_test = X_train/255.0, X_test/255.0

# Add noise
noise = 0.3
X_train_noisy = np.clip(X_train + noise*np.random.randn(*X_train.shape),0,1)
X_test_noisy = np.clip(X_test + noise*np.random.randn(*X_test.shape),0,1)

# Autoencoder model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,3,strides=2,padding='same',
                           input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(8,3,strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(8,3,strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(1,3,strides=2,
                                    activation='sigmoid',
                                    padding='same')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy')

# Train
model.fit(X_train_noisy.reshape(-1,28,28,1),
          X_train.reshape(-1,28,28,1),
          epochs=10,
          batch_size=200)

# Predict
pred = model.predict(X_test_noisy[:5].reshape(-1,28,28,1))

# Display
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test_noisy[i], cmap='gray')
    plt.axis('off')

    plt.subplot(2,5,i+6)
    plt.imshow(pred[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.show()