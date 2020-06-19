from dataset import data_processing
from classify import classify


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_processing()
    k = 50
    test_result = classify(train_data, train_label, test_data, k)
    num = len(test_label)
    temp = 0
    for i in range(num):
        if test_result[i] == test_label[i]:
            temp += 1
    accuracy = temp / num
    print("准确率为: ", accuracy)
