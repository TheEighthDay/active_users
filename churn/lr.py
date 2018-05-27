#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : lr.py
# @Author: JohnHuiWB
# @Date  : 2018/5/27 0027
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from active_users.churn.preprocess import load_preprocessed_data

seed = 1

def lr():
    x, y = load_preprocessed_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    model = Pipeline([
                        # ('sc', MinMaxScaler(feature_range=(0, 1), copy=True)),
                        ('sc', StandardScaler()),
                        ('pca', PCA(random_state=seed)),
                        ('clf', LogisticRegression(random_state=seed))
                        ])
    model.fit(x_train, y_train)
    # print('Test accuracy: %.3f' % model.score(x_test, y_test))

    result = model.predict(x_test)
    # print(result)
    # print(y_test)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(result)):
        if result[i] > 0.5:
            if y_test[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_test[i] == 0:
                tn += 1
            else:
                fn += 1

    print('total:\t\t', (tp + tn + fn + fp))
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    print('Accuracy:\t', accuracy)
    print('Precision:\t', p)
    print('Recall:\t\t', r)
    print('F1-score:\t', f1_score)

if __name__ == '__main__':
    lr()
