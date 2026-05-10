import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

# Select 2 classes: Pullover(2) and T-shirt(0)
mask = (y_train==0) | (y_train==2)

X2, y2 = X_train[mask], y_train[mask]
y2 = (y2==2).astype(int)

# Split data
Xtr, Xval, ytr, yval = train_test_split(X2, y2, test_size=0.2)

# Pretrained model
base_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu')
])

# Add output layer
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Freeze pretrained layers
base_model.trainable = False

# Compile and train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(Xtr, ytr, epochs=5, validation_data=(Xval,yval))

# Evaluate
model.evaluate(Xval, yval)