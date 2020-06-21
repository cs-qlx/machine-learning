def data_read():
    file = open('data.txt')
    lines = file.readlines()
    num = len(lines)
    dataset = []
    label = []
    for i in range(num):
        str1 = lines[i].strip().split('\t')
        dataset.append(str1)
    label = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataset, label
