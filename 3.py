import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop

# Load dataset
data = pd.read_csv('winequality-red.csv', sep=';')

# Split data
X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Optimizers
optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'RMSProp': RMSprop()
}

# Test each optimizer
for name, opt in optimizers.items():

    model = Sequential([
        Dense(64, activation='relu',
              input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"{name} Accuracy: {acc:.4f}")