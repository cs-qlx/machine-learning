from os import listdir
import numpy as np


def file_format(file_name):
    format_data = np.zeros((1, 1024))
    file = open(file_name)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            format_data[0, 32 * i + j] = int(line[j])
    return format_data


def data_processing():
    train_file = listdir('./trainingDigits')
    test_file = listdir('./testDigits')
    len_train = len(train_file)
    len_test = len(test_file)
    train_label = []
    test_label = []
    train_data = np.zeros((len_train, 1024))
    test_data = np.zeros((len_test, 1024))
    for i in range(len_train):
        file_name = train_file[i]
        label = int(file_name.split('_')[0])
        train_label.append(label)
        train_data[i, :] = file_format('./trainingDigits/' + file_name)
    for j in range(len_test):
        file_name = test_file[j]
        label = int(file_name.split('_')[0])
        test_label.append(label)
        test_data[j, :] = file_format('./testDigits/' + file_name)
    return train_data, train_label, test_data, test_label
