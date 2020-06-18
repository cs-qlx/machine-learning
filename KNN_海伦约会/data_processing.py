import numpy as np


def data_norm(data):
    num = data.shape[0]
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    area = data_max - data_min
    for i in range(num):
        data[i, :] = (data[i, :] - data_min) / area
    return data
