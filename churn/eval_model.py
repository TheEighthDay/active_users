#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : eval_model.py
# @Author: JohnHuiWB
# @Date  : 2018/5/27 0027
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from active_users.churn.preprocess import load_preprocessed_data_23_7

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def metrics(result, y, threshold=0.5):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(result)):
        if result[i] > threshold:
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 0:
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


def eval_model():
    x, y = load_preprocessed_data_23_7()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)

    pipelines = dict()
    pipelines['ScalerLR'] = Pipeline(
        [('Scaler', StandardScaler()), ('LR', LogisticRegression(random_state=seed))])
    pipelines['ScalerLinearR'] = Pipeline(
        [('Scaler', StandardScaler()), ('LR', LinearRegression())])
    pipelines['ScalerLASSO'] = Pipeline(
        [('Scaler', StandardScaler()), ('LASSO', Lasso(random_state=seed))])
    pipelines['ScalerEN'] = Pipeline(
        [('Scaler', StandardScaler()), ('EN', ElasticNet(random_state=seed))])
    # pipelines['ScalerKNN'] = Pipeline(
    #     [('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])
    pipelines['ScalerCART'] = Pipeline(
        [('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor(random_state=seed))])
    pipelines['ScalerSVM'] = Pipeline(
        [('Scaler', StandardScaler()), ('SVM', SVR())])
    results = []
    for key in pipelines:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_result = cross_val_score(
            pipelines[key],
            x_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=4)
        results.append(cv_result)
        print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))


def eval_ensemble_model():
    x, y = load_preprocessed_data_23_7()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)

    ensembles = dict()
    ensembles['ScaledAB'] = Pipeline(
        [('Scaler', StandardScaler()), ('AB', AdaBoostRegressor(random_state=seed))])
    # ensembles['ScaledAB-KNN'] = Pipeline([('Scaler',
    #                                        StandardScaler()),
    #                                       ('ABKNN',
    #                                        AdaBoostRegressor(KNeighborsRegressor(),
    #                                                          random_state=seed))])
    ensembles['ScaledAB-LR'] = Pipeline([('Scaler',
                                          StandardScaler()),
                                         ('ABLR',
                                          AdaBoostRegressor(LinearRegression(),
                                                            random_state=seed))])
    ensembles['ScaledRFR'] = Pipeline(
        [('Scaler', StandardScaler()), ('RFR', RandomForestRegressor(random_state=seed))])
    ensembles['ScaledETR'] = Pipeline(
        [('Scaler', StandardScaler()), ('ETR', ExtraTreesRegressor(random_state=seed))])
    ensembles['ScaledGBR'] = Pipeline([('Scaler', StandardScaler(
    )), ('RBR', GradientBoostingRegressor(random_state=seed))])

    results = []
    for key in ensembles:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_result = cross_val_score(
            ensembles[key],
            x_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=4)
        results.append(cv_result)
        print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))


def ada():
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    model = Pipeline([('Scaler', StandardScaler()),
                      ('SVM', AdaBoostRegressor(random_state=seed))])
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    total:		 7490
    Accuracy:	 0.8012016021361815
    Precision:	 0.8025069637883009
    Recall:		 0.7869434580715652
    F1-score:	 0.7946490139291132
    '''


def svr():
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    model = Pipeline([('Scaler', StandardScaler()),
                      ('SVM', SVR(verbose=True))])
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    total:		 7490
    Accuracy:	 0.8080106809078772
    Precision:	 0.8191214470284238
    Recall:		 0.7792952745151598
    F1-score:	 0.7987122060470325
    '''


def linear_r():
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    model = Pipeline([('Scaler', StandardScaler()),
                      ('LR', LinearRegression())])
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    total:		 7490
    Accuracy:	 0.805607476635514
    Precision:	 0.7970897332255457
    Recall:		 0.8079759628516798
    F1-score:	 0.8024959305480195
    '''


def knn():
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    model = Pipeline([('Scaler', StandardScaler()),
                      ('KNN', KNeighborsRegressor())])
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    n_neighbors=5
    total:		 7490
    Accuracy:	 0.7798397863818425
    Precision:	 0.7782079646017699
    Recall:		 0.768642447418738
    F1-score:	 0.7733956300673355
    '''


def gbr():
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    model = Pipeline([('Scaler', StandardScaler()),
                      ('pca', PCA(random_state=seed)),
                      ('RBR', GradientBoostingRegressor(random_state=seed))])
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    total:		 7490
    Accuracy:	 0.8134846461949266
    Precision:	 0.8290697674418605
    Recall:		 0.7790221251024311
    F1-score:	 0.8032671454724687
    '''


def cnn():
    from keras.preprocessing.sequence import pad_sequences
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    from keras.layers import Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, \
        GlobalAveragePooling1D, BatchNormalization
    x, y = load_preprocessed_data_23_7()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    scaler.fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = x_train.reshape(x_train.shape[0], 7, 7, 1)
    x_test = x_test.reshape(x_test.shape[0], 7, 7, 1)

    model = Sequential()

    # model.add(BatchNormalization(input_shape=(7, 7, 1)))
    model.add(Conv2D(32, (3, 3), input_shape=(7, 7, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('begin')
    model.fit(x_train, y_train, batch_size=30, epochs=10, validation_split=0.2, shuffle=True)
    result = model.predict(x_test)
    metrics(result, y_test)

    '''
    total:		 7490
    Accuracy:	 0.8108144192256341
    Precision:	 0.8183881952326901
    Recall:		 0.7877629063097514
    F1-score:	 0.8027835768963117
    '''


if __name__ == '__main__':
    # eval_model()
    # linear_r()
    # svr()
    # knn()
    # eval_ensemble_model()
    # gbr()
    # cnn()
    ada()
