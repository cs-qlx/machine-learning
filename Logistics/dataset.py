
def read_data():
    with open('horseColicTraining.txt') as f:
        train_data = []
        train_label = []
        lines = f.readlines()
        num_train = len(lines)
        for i in range(num_train):
            data = lines[i].strip().split('\t')
            data = list(map(float, data))
            train_data.append(data[0:-2])
            train_label.append(data[-1])
    with open('horseColicTest.txt') as f:
        test_data = []
        test_label = []
        lines = f.readlines()
        num_test = len(lines)
        for i in range(num_test):
            data = lines[i].strip().split('\t')
            data = list(map(float, data))
            test_data.append(data[0:-2])
            test_label.append(data[-1])
    return train_data, train_label, test_data, test_label