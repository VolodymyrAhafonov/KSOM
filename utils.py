import csv
import numpy as np


def read_from_csv(csv_name):
    classes = {}
    data = []
    with open(csv_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            arr = row[0:-1]
            if row[-1] not in classes:
                lenth = len(classes.keys())
                classes[row[- 1]] = lenth
            arr.append(classes[row[-1]])
            data.append(arr)
    data_np = np.array(data, dtype=np.float32)
    attributes = data_np[:, :-1]
    target = data_np[:, -1]
    return attributes, target, classes


def normalize(data):
    normalized_data = np.empty([0, data.shape[1]], np.float32)
    for data_row in data:
        data_normalized_row = data_row / np.sqrt(np.sum(data_row ** 2))
        normalized_data = np.vstack((normalized_data, data_normalized_row))

    return normalized_data
