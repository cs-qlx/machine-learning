import numpy as np


def file_read():
    file = 'hailun.csv'
    fr = open(file)
    lines = fr.readlines()
    num_lines = len(lines)
    test_num = int(0.2 * num_lines)
    data = []
    label = []
    i = 0
    for line in lines:
        if i != 0:
            line = line.strip().split(',')
            line = list(map(float, line))
            data.append(line[0:3])
            label.append(line[-1])
        i += 1
    test_data = data[-test_num:-1].copy()
    test_label = label[-test_num:-1]
    train_data = data[0:num_lines - test_num]
    train_label = label[0:num_lines - test_num]
    train_data = np.mat(train_data)
    test_data = np.mat(test_data)
    fr.close()
    return train_data, train_label, test_data, test_label

