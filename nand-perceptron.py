#!/usr/bin/env python3

"""nand-perceptron.py: 2-input Perceptron that mimics a NAND gate."""

from random import choice

import numpy as np

'''
Implementing NAND Gate Perceptron from: Michael A. Nielsen, "Neural Networks and Deep Learning, Determination Press, 2015"
http://neuralnetworksanddeeplearning.com/chap1.html

'''

'''
Perceptron properties
'''
# "p" for "perception" = W dot X. "Bias" is set to -3.
activation = lambda p: 0 if p + 3 <= 0 else 1
# Start with random weights
W = np.random.rand(2)
learning_rate = 0.1
# Making a standalone learning function to try and make things cleaner.
learn = lambda error, X: learning_rate * error * X

'''
Learning framework
'''
# np.array() is a more efficient data structure - good to get in the habit of using it.

training_data = [(np.array([0, 0]), 1),
                 (np.array([0, 1]), 0),
                 (np.array([1, 0]), 0),
                 (np.array([1, 1]), 0),
                 ]

for i in range(200):
    X, y = choice(training_data)

    p = np.dot(W, X)
    y_pred = activation(p)

    error = y - y_pred
    update = learn(error, X)
    W += update

    # Output after testing
    print("Final predictions:")
    for X, y in training_data:
        p = np.dot(W, X)
        y_pred = activation(p)

        print("X: {}\n    y: {}\n    y_pred: {}\n".format(X, y, y_pred))
