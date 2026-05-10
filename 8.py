#Build an LSTM-based model for time-series forecasting or text generation.


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Time series data
data = [1,2,3,4,5,6,7,8,9,10]

X = []
y = []

# Create sequences
for i in range(len(data)-3):
    X.append(data[i:i+3])
    y.append(data[i+3])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([

    LSTM(50, input_shape=(3,1)),
    Dense(1)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Predict next value
test = np.array([[8,9,10]])
test = test.reshape((1,3,1))

prediction = model.predict(test)

print("Predicted value:", prediction[0][0])