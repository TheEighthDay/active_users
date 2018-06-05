#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : to_submission.py
# @Author: JohnHuiWB
# @Date  : 2018/5/30 0030
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm
import sys
sys.path.append("../")
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.vectorizer import vec
from util.load_data import load_data



def gen_vec_data():
    if os.path.exists('../original_data/x_30.npy'):
        x = np.load('../original_data/x_30.npy')
        ids = np.load('../original_data/id_30.npy')
        print(np.shape(x))
        return x, ids
    else:
        r, l, c, a = load_data()
        x = []
        author_id = list(a['author_id'].get_values())
        for index in tqdm(r.index):
            user_id = r.loc[index]['user_id']
            v = vec(
                r.loc[index].get_values(),
                l.loc[l.user_id == user_id].get_values(),
                c.loc[c.user_id == user_id].get_values(),
                a.loc[a.user_id == user_id].get_values(),
                author_id,
                31)
            x.append(v)
        x = np.array(x)
        ids = r['user_id'].get_values()
        np.save('../original_data/x_30', x)
        np.save('../original_data/id_30', ids)
        print(np.shape(x))
        return x, ids


def to_submission(model_name: str, model, x, ids):
    """

    :param model_name: 模型的名称：字符串
    :param model: 模型，需要有predict方法
    :param x: 经过处理后，可以直接传入predict方法的数据
    :param ids: 与x匹配的user_id
    :return:
    """
    result = model.predict(x)
    ending = []
    for i in range(len(result)):
        if result[i] >= 0.4:
            ending.append(ids[i])

    timestamp = str(time.strftime('_%m_%d_%H_%M', time.localtime()))
    pd.DataFrame(np.array(ending).reshape(-1, 1)
                 )[0].to_csv('../result/'+model_name+timestamp+'.csv', index=False)


def xgb_predict(fn):
    import xgboost as xgb
    bst = xgb.Booster()  # init model
    bst.load_model(fn)  # load data
    x, ids = gen_vec_data()
    x = xgb.DMatrix(x)
    to_submission('xgb', bst, x, ids)
# def lgb_predict(fn):
#     from lightgbm.sklearn import LGBMClassifier
#     from sklearn.externals import joblib
#     bst = joblib.load(fn)
#     x, ids = gen_vec_data()
#     to_submission('lgb', bst, x, ids)


if __name__ == '__main__':
    #xgb_predict('../model/604xgb.model')
    xgb_predict('../model/605xgb.model')