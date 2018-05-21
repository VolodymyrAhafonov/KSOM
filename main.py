import numpy as np
import csv
from neuron import Neuron
from utils import read_from_csv, normalize
from som import Som

n1, n2, n3, = read_from_csv('Iris.csv')
norm_data = normalize(n1)

som = Som(10, norm_data)
som.fit(100, 1e-2)

pass

