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
from sklearn.grid_search import GridSearchCV
from preprocess import load_preprocessed, load_preprocessed_data_23_7

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
windows = [
    # [1, 4, 5, 5],
    # [2, 10, 11, 12],
    # [3, 15, 16, 18],
    # [4, 20, 21, 24],
    # [5, 25, 26, 30],
    [1, 24, 25, 30]
]


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


def eval_model(x, y):
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


def eval_ensemble_model(x, y):
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


def ada(x, y):
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


def svr(x, y):
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


def linear_r(x, y):
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


def knn(x, y):
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


def cnn(x, y):
    from keras.callbacks import ModelCheckpoint
    from keras.preprocessing.sequence import pad_sequences
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import Sequential
    from keras.layers import Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, \
        GlobalAveragePooling1D, BatchNormalization
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, shuffle=True)
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

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(
        '../model/530cnn.model', monitor='val_loss', save_best_only=True)

    model.fit(x_train, y_train, batch_size=30, epochs=30, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


    '''
    total:		 7490
    Accuracy:	 0.8108144192256341
    Precision:	 0.8183881952326901
    Recall:		 0.7877629063097514
    F1-score:	 0.8027835768963117
    '''


def cnn_metrics(x, y):
    from keras.models import load_model

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, shuffle=True)
    scaler = StandardScaler()
    scaler.fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = x_train.reshape(x_train.shape[0], 7, 7, 1)
    x_test = x_test.reshape(x_test.shape[0], 7, 7, 1)

    model = load_model('../model/530cnn.model')
    result = model.predict(x_test)
    metrics(result, y_test)


def xgb(x, y):
    import xgboost as xgb
    from xgboost import plot_importance
    import matplotlib.pyplot as plt

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_test = xgb.DMatrix(x_test)

    # params = {'booster': 'gbtree',
    #           'objective': 'binary:logistic',
    #           'eval_metric': 'auc',
    #           'max_depth': 6,
    #           'lambda': 10,
    #           'subsample': 0.75,
    #           'colsample_bytree': 0.75,
    #           'min_child_weight': 4,
    #           'eta': 0.025,
    #           'seed': 0,
    #           'nthread': 8,
    #           'silent': 1}

    #params={'max_depth':4, 'min_child_weight':6,'eta':0.01, 'silent':0, 'objective':'binary:logistic','subsample':1.0, 'colsample_bytree':1.0, 'lambda': 3, 'alpha':0.2, 'eval_metric':'auc'}
    # 31号晚params={'max_depth':4, 'eta':0.015, 'silent':0, 'objective':'binary:logistic', 'lambda': 4, 'alpha':0.5, 'eval_metric':'auc'}
    params={'eta':0.01, 'n_estimators':140, 'max_depth':4, 'min_child_weight':4, 'gamma':0, 'subsample':0.6, 'colsample_bytree':0.7,'objective':'binary:logistic', 'nthread':4, 'eval_metric':'auc'}
    watchlist = [(d_train, 'train')]
    bst = xgb.train(params, d_train, num_boost_round=400, evals=watchlist)

    result = bst.predict(d_test)
    bst.save_model('../model/531xgb2.model')
    print(result)
    print(y_test)
    metrics(result, y_test)
    # ypred_contribs = bst.predict(d_test, pred_contribs=True)
    # for i in range(len(ypred_contribs[0])):
    #     print(str(i) + ':' + str(ypred_contribs[0][i]))
    plot_importance(bst)
    plt.show()


    '''
    params={'max_depth':5, 'eta':0.01, 'silent':0, 'objective':'binary:logistic', 'lambda': 3, 'alpha':0.2, 'eval_metric':'auc'}
    total:		 8104
    Accuracy:	 0.8257650542941757
    Precision:	 0.8277227722772277
    Recall:		 0.7846607669616519
    F1-score:	 0.8056167400881057
    '''
    '''
    params={'max_depth':4, 'min_child_weight':6,'eta':0.01, 'silent':0, 'objective':'binary:logistic', 'lambda': 3, 'alpha':0.2, 'eval_metric':'auc'}
    total:		 8104
    Accuracy:	 0.8256416584402764
    Precision:	 0.8256467941507312
    Recall:		 0.7873424510592653
    F1-score:	 0.8060398078242965
    time: 5_31_12_30
    '''

def xgb_gridsearch(x, y):
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    from sklearn.grid_search import GridSearchCV
    import numpy as np

    y = np.squeeze(y)
    x = np.squeeze(x)

    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 9)],
        'colsample_bytree': [i / 10.0 for i in range(6, 9)]
    }
    #param_test6 = {
    #    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    #}    正则优化（暂无必要）

    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, max_depth=4,
                                                    min_child_weight=6, gamma=0, subsample=1.0, colsample_bytree=1.0,
                                                    objective='binary:logistic'),
                            param_grid=param_test4, scoring='f1', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(x, y)

    print("Best parameters set found on development set:")
    print(gsearch1.best_params_)

    '''Best parameters set found on development set:
{'max_depth': 4, 'min_child_weight': 4}
{'gamma': 0.0}
{'colsample_bytree': 0.7, 'subsample': 0.6}'''




if __name__ == '__main__':
    # x, y = load_preprocessed_data_23_7()
    x, y = load_preprocessed(windows)
    # eval_model(x, y)
    # linear_r(x, y)
    # svr(x, y)
    # knn(x, y)
    # eval_ensemble_model(x, y)
    # gbr(x, y)
    # cnn(x, y)
    # cnn_metrics(x, y)
    # ada(x, y)
    xgb(x, y)
    #xgb_gridsearch(x, y)
