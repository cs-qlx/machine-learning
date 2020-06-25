from dataset import read_data
import numpy as np
import random
from math import exp


def sigmod(vec, _weight):
    result = np.sum(vec * _weight.T)
    result = 1 / (1 + exp(-result))
    return result


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
    result = sum(x, 1)
    num = data.shape[0]
    for i in range(num):
        temp = result[i] + label[i]
        if temp >= 1.5 or temp < 0.5:
            a += 1
    return a / num


if __name__ == '__main__':
    traindata, trainlabel, testdata, testlabel = read_data()
    weight = train(np.mat(traindata), np.mat(trainlabel).T, 100)
    print("准确率为: ", test(np.array(testdata), np.array(weight), testlabel))
