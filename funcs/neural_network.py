import numpy as np
from scipy import optimize


class Trainer(object):
    def __init__(self, nn):
        self.NN = nn

    def cost_function_wrapper(self, params, X, y):
        self.NN.set_param_from_vector(params)
        cost = self.NN.cost_function(X, y)
        grad = self.NN.compute_gradients(X,y)
        return cost, grad

    def callback_bfgs(self, params):
        self.NN.set_param_from_vector(params)
        self.J.append(self.NN.cost_function(self.X, self.y))
        self.J_test.append(self.NN.cost_function(self.X_test, self.y_test))

    def train(self, train_X, train_y, test_X, test_y):
        self.X = train_X
        self.y = train_y

        self.X_test = test_X
        self.y_test = test_y

        # List to store costs
        self.J = []
        self.J_test = []

        init_params = self.NN.get_param_vector()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, init_params, jac=True, method='BFGS', \
                                 args=(train_X, train_y), options=options, callback=self.callback_bfgs)


class NeuralNetwork(object):
    def __init__(self, count_hidden_layers, ILS, HLS, OLS, a_lambda, af_hl, af_hl_prime, af_ol, af_ol_prime):
        # Define Parameters of network:
        # Number of hidden layers
        self.hl_count = count_hidden_layers
        # Input layer size
        self.ILS = ILS
        # Hidden layer size -> here we assume that all layers have the same number of neurons
        self.HLS = HLS
        # Output layer size
        self.OLS = OLS
        # Lambda for regularization of cost function
        self.Lambda = a_lambda
        # Activation function on hidden layer
        self.af_hl = af_hl
        # Activation function derivative on hidden layer
        self.af_hl_prime = af_hl_prime
        # Activation function on output layer
        self.af_ol = af_ol
        # Activation function derivative on output layer
        self.af_ol_prime = af_ol_prime

        # Weights
        self.weights = []
        for n in range(0, self.hl_count + 1, 1):
            if n == 0:
                self.weights.append(np.random.randn(self.ILS, self.HLS))
            elif n == self.hl_count:
                self.weights.append(np.random.randn(self.HLS, self.OLS))
            else:
                self.weights.append(np.random.randn(self.HLS, self.HLS))


    # Cost function (errors)
    def cost_function(self,X,y):
        self.y_hat = self.forward(X)
        # Cost with regularization
        reg=0.0
        for W in self.weights:
            reg = reg + np.sum(W**2)

        J = 0.5 * sum((y - self.y_hat) ** 2) / X.shape[0] + (self.Lambda / 2) * reg
        return J

    # Derivative from cost function (to reduce errors)
    def cost_function_prime(self, X, y):
        self.y_hat = self.forward(X)

        dJdWn = []
        delta_n = np.multiply(-(y - self.y_hat), self.af_ol_prime(self.Z[self.hl_count]))
        for n in range(self.hl_count+1, 0, -1):
            # Apply for the last the function on output layer
            if n < self.hl_count+1:
                delta_n = np.dot(delta_n, self.weights[n].T) * self.af_hl_prime(self.Z[n-1])
            dJdWn.append(np.dot(self.A[n-1].T, delta_n)/X.shape[0] + self.Lambda*self.weights[n-1])

        return dJdWn[::-1]

    def forward(self, X):
        self.Z = []
        self.A = []
        self.A.append(X)

        for n in range(0, self.hl_count+1, 1):
            self.Z.append(np.dot(self.A[n], self.weights[n]))
            # If the last W -> apply activation function on output layer
            # otherwise apply function on hidden layer
            if n == self.hl_count:
                self.A.append(self.af_ol(self.Z[n]))
                break
            else:
                self.A.append(self.af_hl(self.Z[n]))
            # self.Z.append(np.dot(X, self.weights[n]))
        self.y_hat = self.A[self.hl_count+1]
        return self.y_hat

#  Some methods to operate with parameters from outside the class
    def get_param_vector(self):
        # Get W1, W2 .. Wn as a vector
        flatten_w = [w.ravel() for w in self.weights]
        return np.concatenate(tuple(flatten_w))

    def set_param_from_vector(self, params):
        # Set W1 and W2 from a vector
        self.weights = []

        for n in range(0, self.hl_count+1, 1):
            if n == 0:
                w_start = 0
                w_end = self.HLS * self.ILS
                self.weights.append(np.reshape(params[w_start:w_end], (self.ILS, self.HLS)))
            elif n == self.hl_count:
                w_start = w_end
                w_end = w_start + self.HLS*self.OLS
                self.weights.append(np.reshape(params[w_start:w_end], (self.HLS, self.OLS)))
            else:
                w_start = w_end
                w_end = w_start + self.HLS**2
                self.weights.append(np.reshape(params[w_start:w_end], (self.HLS, self.HLS)))

    def compute_gradients(self, X,y):
        dJdWn = self.cost_function_prime(X,y)
        flatten_dJdWn = []
        for dJdW in dJdWn:
            flatten_dJdWn.append(dJdW.ravel())
        # flatten_dJdWn = [djdW.ravel() for djdW in dJdWn]
        return np.concatenate(tuple(flatten_dJdWn))


# We'll check whether something went wrong
def compute_numerical_gradients(NN, X, y):
    init_params = NN.get_param_vector()
    numerical_grad = np.zeros(init_params.shape)
    delta_X = np.zeros(init_params.shape)
    epsilon = 1e-4

    for n in range(len(init_params)):
        delta_X[n] = epsilon
        NN.set_param_from_vector(init_params+delta_X)
        loss2 = NN.cost_function(X,y)

        NN.set_param_from_vector(init_params-delta_X)
        loss1 = NN.cost_function(X,y)

        numerical_grad[n] = (loss2 - loss1)/(2*epsilon)
        # Initialize delta_X back (for next element in vector!)
        delta_X[n] = 0

    NN.set_param_from_vector(init_params)
    return numerical_grad
