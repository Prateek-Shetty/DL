#Perform basic tensor operations (like addition, multiplication) using Tensor Flow.

import tensorflow as tf

# Create tensors
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Addition
print(tf.add(a, b).numpy())

# Subtraction
print(tf.subtract(a, b).numpy())

# Multiplication
print(tf.multiply(a, b).numpy())

# Division
print(tf.divide(a, b).numpy())

# Square
print(tf.square(a).numpy())

# Broadcasting
print((a + 5).numpy())

# Reshape
print(tf.reshape(a, (4,1)).numpy())

# Concatenate
print(tf.concat([a, b], axis=0).numpy())

# Maximum
print(tf.maximum(a, b).numpy())

# Minimum
print(tf.minimum(a, b).numpy())

# Absolute value
c = tf.constant([[-1, -2], [3, -4]])

print(tf.abs(c).numpy())

# Logarithm
d = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(tf.math.log(d).numpy())

# Exponential
print(tf.exp(d).numpy())