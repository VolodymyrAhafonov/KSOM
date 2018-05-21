import numpy as np


class Neuron:
    weights = np.array(0, dtype=np.float32)
    clastered_examples = []

    def __init__(self, shape):
        self.weights = 2 * np.random.rand(1, shape) - 1

