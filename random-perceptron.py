#!/usr/bin/env python3

"""random-perceptron.py: 3-input Perceptron. Starts with random inputs."""

from random import choice

import numpy as np

'''
Based on David Fumo's script from 'Build a Neural Network from Scratch - Part 2',
https://towardsdatascience.com/build-neural-network-from-scratch-part-2-673ec7cdd89f
'''

np.random.seed(1)  # Set seed to replicate results.

# Function to determine if perceptron "fires". "p" for "perception" = sigma_i(input_i * weight_i) =
# X dot W. Bias is set to 0
activation = lambda p: 0 if p < 0 else 1

training_data = [(np.array([0, 0, 1]), 0),
                 (np.array([0, 1, 1]), 0),
                 (np.array([1, 0, 1]), 0),
                 (np.array([1, 1, 1]), 1)
                 ]

# Training parameters
learning_rate = 0.2
training_steps = 100

# Initialize weights - random.
W = np.random.rand(3)

for i in range(training_steps):
    # Choose a random data point from the training data.
    X, y = choice(training_data)

    # Get the input value and run it through the activation function (effectively the perceptron)
    p = np.dot(W, X)
    y_pred = activation(p)

    # Change the weights by a factor of the error (-1 OR 0 OR 1) * learning rate * [x values] from the step
    error = y - y_pred
    update = learning_rate * error * X
    W += update

    # Output after testing
    print("Predictions after training")

    for X, y in training_data:
        p = np.dot(W, X)
        y_pred = activation(p)

        print("X: {}\n    y: {}\n    y_pred: {}\n".format(X, y, y_pred))
