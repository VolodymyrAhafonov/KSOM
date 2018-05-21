import numpy as np
from utils import normalize


class Neuron:
    weights = np.array(0, dtype=np.float32)

    def __init__(self, shape):
        self.weights = np.random.rand(1, shape)
        self.weights = normalize(self.weights)

    def calculate_distace_to_neuron(self, example):
        return self.weights.dot(example)

    def learn(self, expected_weights, lr_rate):
        self.weights = np.add(self.weights, np.multiply(np.subtract(expected_weights, self.weights), lr_rate))
        self.weights = normalize(self.weights)
