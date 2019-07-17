from sklearn.datasets import fetch_20newsgroups

data_train = fetch_20newsgroups(subset='train',
                                shuffle=True, random_state=42,
                                )

data_test = fetch_20newsgroups(subset='test',
                               shuffle=True, random_state=42,
                             )
# print(data_train.data)
# print(50*'=')
# print(data_test.data)