import numpy as np


class Softmax:
    def __init__(self):
        # No parameters to initialize for Softmax

        # Store input and output for backward pass
        self.input = None
        self.output = None

    def forward(self, x):
        # Save input for backward pass
        self.input = x

        # Calculate the softmax function
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # Save output for backward pass
        self.output = softmax_output

        return softmax_output

    def backward(self, dz):
        # Softmax backward pass
        # The derivative of the softmax function is a Jacobian matrix
        # J[i, j] = softmax_output[i] * (kronecker_delta(i, j) - softmax_output[j])

        batch_size, num_classes = self.input.shape

        # Reshape the 1D array to a 2D array with batch_size rows
        softmax_output = self.output.reshape(batch_size, num_classes)

        # Compute the Jacobian matrix
        jacobian_matrix = -softmax_output[:, :, np.newaxis] * softmax_output[:, np.newaxis, :]

        row_indices = np.arange(batch_size)
        jacobian_matrix[row_indices, :, :] += softmax_output[:, :, np.newaxis]

        # Multiply the Jacobian matrix by the upstream gradient (dz)
        dx = np.einsum('ijk,ik->ij', jacobian_matrix, dz)

        return dx
