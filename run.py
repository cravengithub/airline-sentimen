import os
import time as tm
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import *
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import  SelectKBest

import airline_util as au

os.system('cls')

porter = PorterStemmer()

# obtain corpus
file = 'train.csv'
#file = 'Tweets.csv'
dataFrame = pd.read_csv(file)
corpus = dataFrame.get('text')
sentiments = np.array(dataFrame.get('airline_sentiment'))

# preprocessing text
clean_text_training_list = []
for text in corpus:
    clean_text_training = ''
    # tokenizer corpus, stopwords removal, and remove letter sign
    for s in au.tokenizer(str(text)):
        # stemming word
        clean_text_training = clean_text_training + porter.stem(s)+ ' '
        # clean_text_training = clean_text_training + s +' '
    clean_text_training_list.append(clean_text_training[0:len(clean_text_training)])

# making bag of word using vectorization
np.set_printoptions(precision=2)
docs = np.array(clean_text_training_list)
count = CountVectorizer()
bag = count.fit_transform(docs)
# print(bag.toarray())


# define label
label = []
for lb in sentiments:
    label.append(lb)

# give weight using tf idf and extract features
tfidf = TfidfTransformer(smooth_idf=False)
features = tfidf.fit_transform(bag)
# print(features.shape)
# mi = mutual_info_classif(features, label
new_feature = SelectKBest(mutual_info_classif, k=10).fit_transform(features, label)
# print(new_feature.toarray())
rst = mutual_info_classif(new_feature, label)
s
for score, fname in sorted(zip(rst, label), reverse=True):
    print(fname, score)
'''

# declare classifier and training data
bayes = MultinomialNB()
gnb = GaussianNB()
clf   = LogisticRegression(solver='lbfgs', multi_class='auto')
C=1.0
svm_rfb = SVC(kernel='rbf', gamma=0.7, C=C)
svm_poly =  SVC(kernel='poly', degree=3,  gamma='auto', C=C)
svm_linear = SVC(kernel='linear', C=C)
tree = DecisionTreeClassifier()
nbrs = NearestNeighbors()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)

# model = bayes.fit(features, np.array(label))

# #evaluation using cross validation
scores = cross_val_score(svm_linear, new_feature, label, cv = 3)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
'''


'''
for i in range(0,101):
    print(clean_text_training_list[i])
print('length: ', len(clean_text_training_list))
'''

'''
kf = KFold(n_splits=3,random_state=42,shuffle=True)
accuracy = []
for train_index, test_index in kf.split(features):
    x_train = features[train_index]
    y_train = label[train_index]
    x_test = features[test_index]
    y_test = label[test_index]

    bayes = MultinomialNB()
    bayes.fit(x_train, y_train)
    predict = bayes.predict(x_test)
    acc = accuracy_score(y_test, predict)
    accuracy.append(acc)
avg = np.mean(accuracy)
print(avg)
'''


'''
print(str(corpus[0]))
rs = au.tokenizer(str(corpus[0]))
print(rs)
'''

'''
#feature selection using mutual information
rst = mutual_info_classif(features, label)
for score, fname in sorted(zip(rst, label), reverse=True):
    print(fname, score)
'''