from neuron import Neuron
import numpy as np
import random


class Som:
    neurons = []
    data = np.empty([1, 1], np.float32)
    labels = np.empty([1, 1], np.uint8)

    def __init__(self, k, data):
        for number in range(0, k):
            self.neurons.append(Neuron(data.shape[1]))
        self.data = data
        self.labels = np.empty([data.shape[0], 1], np.uint8)

    def find_closest_neuron(self, example):
        max_distance = -2
        index = 0
        for i in range(0, len(self.neurons)):
            distance = self.neurons[i].calculate_distace_to_neuron(example)
            if distance > max_distance:
                max_distance = distance
                index = i
        return index

    def fit(self, iterations, lr_rate):
        data = np.append(self.data, self.labels, axis=1)
        for iteration in range(0, iterations):
            #random.shuffle(data)
            for example in data:
                index = self.find_closest_neuron(example[:-1])
                self.neurons[index].learn(example[:-1], lr_rate)
                example[-1] = index
        self.data = data[:, :-1]
        self.labels = data[:, -1]
