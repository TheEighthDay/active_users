#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : preprocess.py
# @Author: JohnHuiWB
# @Date  : 2018/5/26 0026
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm
import sys
sys.path.append("../../")
from tqdm import tqdm
import numpy as np
from active_users.util.load_data import load_data
from active_users.util.vectorizer import vec


def show_distribution():
    r, l, c, a = load_data()

    # print('register_user_id count')
    # print(len(r))
    #
    # print('video_id count')
    # print(len(a['video_id'].drop_duplicates()))
    #
    # print('author_id count')
    # print(len(a['author_id'].drop_duplicates()))
    #
    # print('page')
    # print('0:', len(a[a.page == 0]))
    # print('1:', len(a[a.page == 1]))
    # print('2:', len(a[a.page == 2]))
    # print('3:', len(a[a.page == 3]))
    # print('4:', len(a[a.page == 4]))
    # print('>=5:', len(a[a.page >= 5]))
    #
    # print('action_type')
    # print('0:', len(a[a.action_type == 0]))
    # print('1:', len(a[a.action_type == 1]))
    # print('2:', len(a[a.action_type == 2]))
    # print('3:', len(a[a.action_type == 3]))
    # print('4:', len(a[a.action_type == 4]))
    # print('5:', len(a[a.action_type == 5]))
    # print('>=6:', len(a[a.action_type >= 6]))

    print(r.describe())
    print(l.describe())
    print(c.describe())
    print(a.describe())


def preprocessing_23_7():
    r, l, c, a = load_data()
    # 将1到23天作为训练集
    x_l = l[l.day < 24]
    x_a = a[a.day < 24]
    x_r = r[r.register_day < 24]
    x_c = c[c.day < 24]
    last_week_l_u = l[l.day >= 24]['user_id'].drop_duplicates().get_values()
    author_id = list(a['author_id'].get_values())

    x, y = [], []

    for index in tqdm(x_r.index):
        user_id = x_r.loc[index]['user_id']
        v = vec(
            x_r.loc[index].get_values(),
            x_l.loc[x_l.user_id == user_id].get_values(),
            x_c.loc[x_c.user_id == user_id].get_values(),
            x_a.loc[x_a.user_id == user_id].get_values(),
            author_id,
            24)

        # 根据最后7天中用户是否登录app，来判断是否都活跃用户，
        # 仅凭判断user_id是否在launch_log中即可，不必判断另外两个log。
        is_active = 1 if user_id in last_week_l_u else 0

        x.append(v)
        y.append(is_active)

    x = np.array(x)
    y = np.array(y)
    np.save('../original_data/x_23_7', x)
    np.save('../original_data/y_23_7', y)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)


def load_preprocessed_data_23_7():
    x = np.load('../original_data/x_23_7.npy')
    y = np.load('../original_data/y_23_7.npy')
    return x, y.astype(np.float32)


def preprocessing_24_6():
    r, l, c, a = load_data()
    # 将1到23天作为训练集
    x_l = l[l.day < 25]
    x_a = a[a.day < 25]
    x_r = r[r.register_day < 25]
    x_c = c[c.day < 25]
    last_week_l_u = l[l.day >= 25]['user_id'].drop_duplicates().get_values()
    author_id = list(a['author_id'].get_values())

    x, y = [], []

    for index in tqdm(x_r.index):
        user_id = x_r.loc[index]['user_id']
        v = vec(
            x_r.loc[index].get_values(),
            x_l.loc[x_l.user_id == user_id].get_values(),
            x_c.loc[x_c.user_id == user_id].get_values(),
            x_a.loc[x_a.user_id == user_id].get_values(),
            author_id,
            25)

        # 根据最后7天中用户是否登录app，来判断是否都活跃用户，
        # 仅凭判断user_id是否在launch_log中即可，不必判断另外两个log。
        is_active = 1 if user_id in last_week_l_u else 0

        x.append(v)
        y.append(is_active)

    x = np.array(x)
    y = np.array(y)
    np.save('../original_data/x_24_6', x)
    np.save('../original_data/y_24_6', y)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)


def load_preprocessed_data_24_6():
    x = np.load('../original_data/x_24_6.npy')
    y = np.load('../original_data/y_24_6.npy')
    return x, y.astype(np.float32)


if __name__ == '__main__':
    # preprocessing_23_7()
    # show_distribution()
    X,y=load_preprocessed_data_23_7()
    print(np.shape(y))
