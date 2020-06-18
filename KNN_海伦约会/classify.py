import numpy as np


def classify(module, label, num, testdata):
    module_len = module.shape[0]
    eig_num = module.shape[1]
    result = []
    target = [0, 0, 0]
    for i in range(module_len):
        cal_sum = 0
        for j in range(eig_num):
            cal_sum += (module[i, j] - testdata[0, j]) ** 2
        result.append(cal_sum)
    for k in range(num):
        pos = result.index(min(result))
        if label[pos] == 1:
            target[0] += 1
        elif label[pos] == 2:
            target[1] += 1
        elif label[pos] == 3:
            target[2] += 1
    return target.index(max(target)) + 1
