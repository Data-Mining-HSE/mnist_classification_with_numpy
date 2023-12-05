import numpy as np


class LinearLayer:
    def __init__(self, input_size, output_size, id):
        # Initialize weights and biases
        self.id = id
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

        # Store input and output for backward pass
        self.input = None
        self.output = None

        # Store gradients for weights and biases
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x):
        # Save input for backward pass
        self.input = x

        # Calculate linear transformation
        z = np.dot(x, self.weights) + self.biases # 1: 64x128 2: 64x10

        # Save output for backward pass
        self.output = z
        return z

    def backward(self, err, _input):
        # Calculate gradients
        self.grad_weights = np.dot(_input.T, err)
        self.grad_biases = np.sum(err, axis=0, keepdims=True)

        # Calculate gradient for the input
        hidden_layer_error = None if self.id == 1 else np.dot(err, self.weights.T)
        return hidden_layer_error
