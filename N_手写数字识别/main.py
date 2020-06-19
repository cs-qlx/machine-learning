from dataset import data_processing
from classify import classify


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_processing()
    k = 100
    test_result = classify(train_data, train_label, test_data, k)
    num = test_label.shape[0]
    temp = 0
    for i in range(num):
        if test_label[i, test_result[i]] == 1:
            temp += 1
    accuracy = temp / num
    print("准确率为: ", accuracy)
