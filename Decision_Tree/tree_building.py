from read_data import data_read
from math import log
import operator
from draw_tree import*


def cal_ShannoEnt(dataset):
    num = len(dataset)
    label_counts = {}
    shannoEnt = 0
    for i in range(num):
        a = dataset[i][-1]
        if a not in list(label_counts.keys()):
            label_counts[a] = 1
        else:
            label_counts[a] += 1
    for key in label_counts:
        temp = label_counts[key] / num
        shannoEnt -= temp * log(temp, 2)
    return shannoEnt


def split_dataset(dataset, feature, value):
    split_data = []
    for vec in dataset:
        if vec[feature] == value:
            temp = vec[:feature]
            temp.extend(vec[feature + 1:])
            split_data.append(temp)
    return split_data


def best_split_feature(dataset):
    base_shannoEnt = cal_ShannoEnt(dataset)
    base_gain = 0
    num_features = len(dataset[0]) - 1
    for i in range(num_features):
        feature_values = []
        for j in range(len(dataset)):
            feature_values.append(dataset[j][i])
        feature_values = set(feature_values)
        for value in feature_values:
            subdataset = split_dataset(dataset, i, value)
            temp = len(subdataset) / len(dataset)
            new_shannoEnt = temp * log(temp, 2)
            info_gain = base_shannoEnt - new_shannoEnt
            if info_gain > base_gain:
                best_gain = info_gain
                best_feature = i
    return best_feature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def create_Tree(dataset, label):
    num = len(dataset)
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    root_feature = best_split_feature(dataset)
    root_label = label[root_feature]
    decision_tree = {root_label: {}}
    del (label[root_feature])
    root_val = [dataset[i][root_feature] for i in range(num)]
    unique_val = set(root_val)
    for val in unique_val:
        sublabels = label[:]
        decision_tree[root_label][val] = create_Tree(split_dataset(dataset, root_feature, val), sublabels)
    return decision_tree


if __name__ == '__main__':
    data, labels = data_read()
    createPlot(create_Tree(data, labels))

