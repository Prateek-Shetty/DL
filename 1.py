#Perform basic tensor operations (like addition, multiplication) using Tensor Flow

import tensorflow as tf
import numpy as np

tensor1 = tf.constant(np.random.randint(1, 10, (2, 2)), dtype=tf.float32)

tensor2 = tf.constant([[-2, 4],
                       [6, 8]], dtype=tf.float32)

print("Tensor 1:\n", tensor1.numpy())
print("\nTensor 2:\n", tensor2.numpy())

print("\nAddition:\n", tf.add(tensor1, tensor2).numpy())

print("\nSubtraction:\n", tf.subtract(tensor1, tensor2).numpy())

print("\nMultiplication:\n", tf.multiply(tensor1, tensor2).numpy())

print("\nDivision:\n", tf.divide(tensor1, tensor2).numpy())

print("\nSquare:\n", tf.square(tensor1).numpy())

print("\nBroadcasting (+5):\n", (tensor1 + 5).numpy())

print("\nReshape:\n", tf.reshape(tensor1, (4, 1)).numpy())

print("\nConcatenation:\n", tf.concat([tensor1, tensor2], axis=0).numpy())

print("\nMaximum:\n", tf.maximum(tensor1, tensor2).numpy())

print("\nMinimum:\n", tf.minimum(tensor1, tensor2).numpy())

print("\nAbsolute Value:\n",
      tf.abs(tensor2).numpy())

print("\nLogarithm:\n", tf.math.log(tensor2).numpy())

print("\nExponential:\n", tf.exp(tensor2).numpy())
