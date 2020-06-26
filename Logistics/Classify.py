from dataset import read_data
import numpy as np
import random
from math import exp


def sigmod(vec, _weight):
    result = np.sum(vec * _weight)
    result = 1 / (1 + exp(-result))
    return result


def data_norm(data):
    num = data.shape[0]
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    area = data_max - data_min
    for i in range(num):
        data[i, :] = (data[i, :] - data_min) / area
    return data


def train(data, label, n):
    weights = np.ones((1, data.shape[1]))
    num = data.shape[0]
    count = list(range(num))
    for i in range(n):
        temp = random .randint(0, len(count) - 1)
        choose = count[temp]
        del(count[temp])
        alpha = 4 / (1 + i + choose) + 0.01
        train_vec = data[choose]
        error = int(label[choose]) - sigmod(train_vec, weights)
        weights = weights + alpha * weights * error * train_vec.T
    return weights


def test(data, weights, label):
    x = data * weights
    a = 0
    result = np.sum(x, axis=1)
    num = data.shape[0]
    for i in range(num):
        temp = 1 / (1 + exp(-result[i])) + label[i]
        if temp >= 1.5 or temp < 0.5:
            a += 1
    return a / num


if __name__ == '__main__':
    traindata, trainlabel, testdata, testlabel = read_data()
    traindata = data_norm(np.array(traindata))
    testdata = data_norm(np.array(testdata))
    trainlabel = np.array(trainlabel)
    testlabel = np. array(testlabel)
    weight = train(traindata, trainlabel, 100)
    print("准确率为: ", test(np.array(testdata), np.array(weight), testlabel))
