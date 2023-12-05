import numpy as np

from nn.activation.relu import ReLU
from nn.activation.softmax import Softmax
from nn.linear_layer import LinearLayer


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Define layers
        self.linear1 = LinearLayer(input_size, hidden_size, id = 1)
        self.relu = ReLU()
        self.linear2 = LinearLayer(hidden_size, output_size, id = 2)
        self.softmax = Softmax()

    def forward(self, x):
        # Forward pass
        self.input = x
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        self.hidden = x
        x = self.linear2.forward(x)
        x = self.softmax.forward(x)
        self.output = x
        return x

    def backward(self, dz):
        dz = self.linear2.backward(dz, self.hidden)
        dz = self.relu.backward(dz)
        dz = self.linear1.backward(dz, self.input)
        return dz

    def get_accuracy(self, X, y):
        # Evaluation
        correct_predictions = 0
        for i in range(len(X)):
            input_data = X[i:i+1]
            target = np.argmax(y[i])
            predicted = np.argmax(self.forward(input_data))
            if target == predicted:
                correct_predictions += 1

        accuracy = correct_predictions / len(X)
        return accuracy
