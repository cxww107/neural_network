import unittest
import numpy as np
from funcs import NeuralNetwork, Trainer, compute_numerical_gradients


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        # Activation function
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Derivative from activation function
        def sigmoid_prime(z):
            return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

        self.nn = NeuralNetwork(count_hidden_layers=1, ILS=2, HLS=3, OLS=1, a_lambda=1e-4, \
                           af_hl=sigmoid, af_hl_prime=sigmoid_prime, \
                           af_ol=sigmoid, af_ol_prime=sigmoid_prime)

        self.trainer = Trainer(self.nn)

    def test_model(self):

        X_train = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
        y_train = np.array(([75], [82], [93], [70]), dtype=float)

        X_test = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
        y_test = np.array(([70], [89], [85], [75]), dtype=float)

        # Scaling the data
        X_train = X_train / np.amax(X_train, axis=0)
        y_train = y_train / 100

        X_test = X_test / np.amax(X_test, axis=0)
        y_test = y_test / 100

        grad = self.nn.compute_gradients(X_train, y_train)
        print(grad)

        numgrad = compute_numerical_gradients(self.nn, X_train, y_train)
        print(numgrad)

        print(np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad))

        self.trainer.train(X_train, y_train, X_test, y_test)

        print(self.nn.forward(X_train))
        print(y_train)

        check_number = (np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad))
        # Check number has to be in order 1E-8
        self.assertLessEqual(check_number, 1e-8)

