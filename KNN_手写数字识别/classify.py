import numpy as np
from collections import Counter


def classify(train_data, train_label, test_data, classify_num):
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]
    result = []
    for x in range(test_num):
        similarity_list = []
        for i in range(train_num):
            cal_similarity = sum((train_data[i, :] - test_data[x, :]) ** 2, 1)
            similarity_list.append(cal_similarity)
        temp = []
        for k in range(classify_num):
            pos = similarity_list.index(min(similarity_list))
            similarity_list[pos] = 9999
            temp.append(train_label[pos])
        counts = Counter(temp)
        result.append(counts.most_common(1)[0][0])
    return result





