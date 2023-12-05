import numpy as np


class ReLU:
    def __init__(self):
        # No parameters to initialize for ReLU

        # Store input for backward pass
        self.input = None

    def forward(self, x):
        # Save input for backward pass
        self.input = x

        # Apply ReLU activation element-wise
        return np.maximum(0, x)

    def backward(self, dz):
        # Apply derivative of ReLU to the input gradient
        dx = dz * (self.input > 0)

        return dx
