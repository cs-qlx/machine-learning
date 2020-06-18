import numpy as np
from read_data import file_read
from data_processing import data_norm
from classify import classify

if __name__ == '__main__':
    data, train_label, testdata, test_label = file_read()
    data = data_norm(data)
    testdata = data_norm(testdata)
    classify_num = 10
    num = testdata.shape[0]
    accuracy = 0
    for i in range(num):
        if classify(data, train_label, classify_num, testdata[i, :]) == test_label[i]:
            accuracy += 1
    accuracy = accuracy / num
    print("准确率为： %f" % accuracy)
