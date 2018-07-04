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
from util.load_data import load_data
from util.vectorizer import vec


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


def preprocessing_sliding(
        feature_window_begin,
        feature_window_end,
        label_window_begin,
        label_window_end):
    """

    :param feature_window_begin:
    :param feature_window_end:
    :param label_window_begin:
    :param label_window_end:
    :return:
    """

    r, l, c, a = load_data()

    x_r = r[r.register_day >= feature_window_begin]
    x_r = x_r[x_r.register_day <= feature_window_end]
    x_l = l[l.day >= feature_window_begin]
    x_l = x_l[x_l.day <= feature_window_end]
    x_c = c[c.day >= feature_window_begin]
    x_c = x_c[x_c.day <= feature_window_end]
    x_a = a[a.day >= feature_window_begin]
    x_a = x_a[x_a.day <= feature_window_end]

    label_window_launch_user_id = l[l.day >= label_window_begin]
    label_window_launch_user_id = label_window_launch_user_id \
        [label_window_launch_user_id.day <= label_window_end] \
        ['user_id'].drop_duplicates().get_values()
    author_id = list(x_a['author_id'].get_values())

    x, y = [], []

    for index in tqdm(x_r.index):
        user_id = x_r.loc[index]['user_id']
        v = vec(
            x_r.loc[index].get_values(),
            x_l.loc[x_l.user_id == user_id].get_values(),
            x_c.loc[x_c.user_id == user_id].get_values(),
            x_a.loc[x_a.user_id == user_id].get_values(),
            author_id,
            label_window_begin)

        # 根据label window中用户是否登录app，来判断是否都活跃用户，
        # 仅凭判断user_id是否在launch_log中即可，不必判断另外两个log。
        is_active = 1 if user_id in label_window_launch_user_id else 0

        x.append(v)
        y.append(is_active)

    x = np.array(x)
    y = np.array(y)
    np.save(
        '../original_data/x_%d_%d_%d_%d' %
        (feature_window_begin,
         feature_window_end,
         label_window_begin,
         label_window_end),
        x)
    np.save(
        '../original_data/y_%d_%d_%d_%d' %
        (feature_window_begin,
         feature_window_end,
         label_window_begin,
         label_window_end),
        y)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)


def load_preprocessed_sliding(
        feature_window_begin,
        feature_window_end,
        label_window_begin,
        label_window_end):
    x = np.load(
        '../original_data/x_%d_%d_%d_%d.npy' %
        (feature_window_begin,
         feature_window_end,
         label_window_begin,
         label_window_end))
    y = np.load(
        '../original_data/y_%d_%d_%d_%d.npy' %
        (feature_window_begin,
         feature_window_end,
         label_window_begin,
         label_window_end))
    return x, y.astype(np.float32)


def load_preprocessed(windows: list):
    """

    :param windows: 传入一个两层的list，读取所有文件，例如：
                    [[1, 4, 5, 5], [2, 10, 11, 12], [3, 15, 16, 18]]
    :return:
    """
    x, y = [], []
    for fwb, fwe, lwb, lwe in windows:
        x_tmp, y_tmp = load_preprocessed_sliding(fwb, fwe, lwb, lwe)
        x.append(x_tmp)
        y.append(y_tmp.reshape((y_tmp.shape[0], 1)))

    return np.row_stack(x), np.row_stack(y)


def preprocessing_all():
    """
    生成所有的特征向量
    :return:
    """
    r, l, c, a = load_data()

    author_id = list(a['author_id'].get_values())

    x = []

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
    np.save('../original_data/x_30', x)
    print('x.shape:', x.shape)


def load_preprocessed_data_23_7():
    x = np.load('../original_data/x_23_7.npy')
    y = np.load('../original_data/y_23_7.npy')

    return x, y.astype(np.float32)


if __name__ == '__main__':
    # show_distribution()

    # preprocessing_sliding(1, 4, 5, 5)
    # preprocessing_sliding(2, 10, 11, 12)
    # preprocessing_sliding(3, 15, 16, 18)
    # preprocessing_sliding(4, 20, 21, 24)
    # preprocessing_sliding(5, 25, 26, 30)
    preprocessing_sliding(1, 24, 25, 30)
    windows = [
        # [1, 4, 5, 5],
        # [2, 10, 11, 12],
        # [3, 15, 16, 18],
        # [4, 20, 21, 24],
        # [5, 25, 26, 30],
        [1, 24, 25, 30]
    ]
    x, y = load_preprocessed(windows)
    print(x.shape)
    print(y.shape)
    # preprocessing_all()