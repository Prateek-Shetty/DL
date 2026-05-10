#Implement a basic RNN for sequence prediction

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate sine wave
data = np.sin(np.linspace(0, 50, 100))

# Prepare dataset
X, y = [], []
for i in range(90):
    X.append(data[i:i+10])
    y.append(data[i+10])

X = np.array(X).reshape(90, 10, 1)
y = np.array(y)

# Build model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(10,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=50, verbose=0)

# Predict
pred = model.predict(X)

print("Actual:", y[:5])
print("Predicted:", pred[:5].flatten())