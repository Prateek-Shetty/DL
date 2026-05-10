import tensorflow as tf
import numpy as np

# Load text
text = open("shakespeare.txt").read().lower()

# Character mapping
chars = sorted(set(text))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}

# Create sequences
seq_len = 50
X, y = [], []

for i in range(len(text)-seq_len):
    X.append([c2i[c] for c in text[i:i+seq_len]])
    y.append(c2i[text[i+seq_len]])

X = np.array(X).reshape(-1, seq_len, 1) / len(chars)
y = np.array(y)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

# Train
model.fit(X, y, epochs=10, batch_size=64)

# Generate text
seed = "shall i compare thee "
generated = seed

for _ in range(100):
    x = np.array([[c2i[c] for c in generated[-seq_len:]]])
    x = x.reshape(1, seq_len, 1) / len(chars)

    pred = np.argmax(model.predict(x, verbose=0))
    generated += i2c[pred]

print(generated)