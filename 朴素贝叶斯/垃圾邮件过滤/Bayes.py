from dataset import read_email
import numpy as np
from math import log


def create_vector(data, word_list):
    num_word = len(word_list)
    result = []
    if isinstance(data[0], list):
        num = len(data)
        for i in range(num):
            num1 = len(data[i])
            vec = [0] * num_word
            for j in range(num1):
                if data[i][j] in word_list:
                    vec[word_list.index(data[i][j])] = 1
            result.append(vec)
        return result
    else:
        num = len(data)
        vec = [0] * num_word
        for i in range(num):
            if data[i] in word_list:
                vec[word_list.index(data[i])] = 1
        return vec


def train_Bayes(text_matrix, text_label):
    num_text = len(text_matrix)
    text_label = np.array(text_label)
    p1_wordvec = np.ones(len(text_matrix[0]))
    p0_wordvec = np.ones(len(text_matrix[0]))
    p0_all_words = 2
    p1_all_words = 2
    text_matrix = np.array(text_matrix)
    for i in range(num_text):
        if text_label[i] == 1:
            p1_wordvec += text_matrix[i]
            p1_all_words += sum(text_matrix[i])
        else:
            p0_wordvec += text_matrix[i]
            p0_all_words += sum(text_matrix[i])
    return p0_wordvec / p0_all_words, p1_wordvec / p1_all_words, sum(text_label) / len(text_label)


def classify(toclassify_vec, p0_vec, p1_vec, p01):
    toclassify_vec = np.array(toclassify_vec)
    x = sum(toclassify_vec * p0_vec) + log(p01)
    y = sum(toclassify_vec * p1_vec) + log(1 - p01)
    if x > y:
        return 0
    else:
        return 1


if __name__ == '__main__':
    traindata, trainlabel, testdata, testlabel, wordtable = read_email()
    train_vec = create_vector(traindata, wordtable)
    p0, p1, p = train_Bayes(train_vec, trainlabel)
    temp = 0
    if isinstance(testdata[0], list):
        test_num = len(testdata)
        test_vec = create_vector(testdata, wordtable)
        for i in range(test_num):
            if classify(test_vec[i], p0, p1, p) == testlabel[i]:
                temp += 1
        print("准确率为: ", temp/test_num)
    else:
        test_num = 1
        test_vec = create_vector(testdata, wordtable)
        if classify(test_vec, p0, p1, p) == testlabel:
            temp += 1
        print("准确率为: ", temp/test_num)
