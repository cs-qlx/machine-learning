from os import listdir
import re
import random


def email_processing(text):
    text = re.split(r'\W', text)
    return [i.lower() for i in text if len(i) > 2]


def read_email():
    good_email = listdir('./email/ham')
    bad_email = listdir('./email/spam')
    num_goodemail = len(good_email)
    num_bademail = len(bad_email)
    all_email = []
    all_word = []
    email_label = []
    test_data = []
    test_data_label =[]
    for i in range(num_goodemail):
        filename = good_email[i]
        with open('./email/ham/' + filename) as f:
            text = f.read()
            text = email_processing(text)
            all_word.extend(text)
            all_email.append(text)
            email_label.append(0)
    for j in range(num_bademail):
        filename = bad_email[j]
        with open('./email/spam/' + filename) as f:
            text = f.read()
            text = email_processing(text)
            all_email.append(text)
            all_word.extend(text)
            email_label.append(1)
    word_table = set(all_word)
    for i in range(10):
        x = random.randint(0, 49 - i)
        test_data.append(all_email[x])
        test_data_label.append(email_label[x])
        del(email_label[x])
        del(all_email[x])
    train_data = all_email
    train_data_label = email_label
    return train_data, train_data_label, test_data, test_data_label, list(word_table)
