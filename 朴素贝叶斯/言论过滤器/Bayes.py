from dataset import read_data
import numpy as np
from math import log


def word_table(data):
    num = len(data)
    table = set([])
    for i in range(num):
        table = table | set(data[i])
    return list(table)


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
    p1_wordvec = np.zeros(len(text_matrix[0]))
    p0_wordvec = np.zeros(len(text_matrix[0]))
    p0_all_words = 0
    p1_all_words = 0
    text_matrix = np.array(text_matrix)
    for i in range(num_text):
        if text_label[i] == 1:
            p1_wordvec += text_matrix[i]
            p1_all_words += sum(text_matrix[i])
        else:
            p0_wordvec += text_matrix[i]
            p0_all_words += sum(text_matrix[i])
    return p0_wordvec / p0_all_words, p1_wordvec / p1_all_words, sum(text_label) / len(text_label)


def classify(toclassify_vec, p0_vec, p1_vec, p1):
    toclassify_vec = np.array(toclassify_vec)
    x = sum(toclassify_vec * p0_vec) + log(p1)
    y = sum(toclassify_vec * p1_vec) + log(1 - p1)
    if x > y:
        return 0
    else:
        return 1


if __name__ == '__main__':
    text, label = read_data()
    wordtable = word_table(text)
    text_mat = create_vector(text, wordtable)
    p0, p1, p = train_Bayes(text_mat, label)
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(create_vector(testEntry, wordtable))  # 测试样本向量化
    if classify(thisDoc, p0, p1, p):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(create_vector(testEntry, wordtable))  # 测试样本向量化
    if classify(thisDoc, p0, p1, p):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')
