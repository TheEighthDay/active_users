#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : vectorizer.py
# @Author: JohnHuiWB
# @Date  : 2018/5/27 0027
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm


import numpy as np
from active_users.util.load_data import load_data


def vec(r: list, l: list, c: list, a: list, author_id: list, now):
    """
    输入一个用户对应的四个log中的数据，处理后返回特征向量
    :param r:
    :param l:
    :param c:
    :param a:
    :param author_id: activity_log中所有author_id
    :param now: 当前时间（天），如用前23天作为训练集是，now应为24
    :return:
    """

    # print(r)
    # print(l)
    # print(c)
    # print(a)

    user_id = r[0]
    register_day = r[1]
    register_type = r[2]
    device_type = r[3]
    # print('user_id:', user_id)
    # print('register_day:', register_day)
    # print('register_type', register_type)
    # print('device_type', device_type)

    register_time = now - register_day
    # print('register_time', register_time)

    launch_time = len(l)  # 登录天数
    # print('launch_time:', launch_time)

    active_ratio = launch_time / register_time
    # print('active_ratio:', active_ratio)

    launch_day_list = sorted([x[1] for x in l])
    no_l_ratio = (now - max(launch_day_list)) / register_time
    # print('no_l_ratio:', no_l_ratio)

    l_last_ratio = consecutive_days(launch_day_list) / register_time
    # print('l_last_ratio:', l_last_ratio)

    active_2_ratio = 0 if int(register_time / 2) == 0 else len([x[1] for x in l if x[1] > (
        now - register_day - int(register_time / 2))]) / int(register_time / 2)
    # print('active_2_ratio', active_2_ratio)
    active_4_ratio = 0 if int(register_time / 4) == 0 else len([x[1] for x in l if x[1] > (
        now - register_day - int(register_time / 4))]) / int(register_time / 4)
    # print('active_4_ratio', active_4_ratio)
    active_8_ratio = 0 if int(register_time / 8) == 0 else len([x[1] for x in l if x[1] > (
        now - register_day - int(register_time / 8))]) / int(register_time / 8)
    # print('active_8_ratio', active_8_ratio)

    create_list = sorted([x[1] for x in c])
    create_num = len(create_list)  # 拍摄次数
    # print('create_list_len:', create_list_len)
    create_day = list(set(create_list))  # 有哪些天进行了拍摄
    # print('create_day:', create_day)
    create_per_d = create_num / register_time
    # print('create_per_d:', create_per_d)

    create_per_l = create_num / launch_time
    # print('create_per_l:', create_per_l)

    create_cnt_list = [create_list.count(x) for x in create_day]  # 每天拍摄的数目
    # print('create_cnt_list:', create_cnt_list)
    create_max = 0 if create_num == 0 else max(create_cnt_list)
    # print('create_max:', create_max)

    create_sd = 0 if len(create_day) <= 1 else np.std(create_cnt_list, ddof=1)
    # print('create_sd:', create_sd)

    create_last = consecutive_days(list(create_day)) / register_time
    # print('create_last:', create_last)

    create_final_day = register_day if create_num == 0 else max(create_day)
    # print('create_final_day:', create_final_day)
    create_final_ratio = (now - create_final_day) / register_time
    # print('create_final_ratio:', create_final_ratio)

    action_list = [x[1] for x in a]
    action_list.sort()
    action_list_len = len(action_list)  # 行为次数
    # print('action_list_len:', action_list_len)
    action_day = list(set(action_list))  # 有哪些天有行为
    # print('action_day:', action_day)
    action_per_d = action_list_len / register_time
    # print('action_per_d:', action_per_d)

    action_per_l = action_list_len / launch_time
    # print('action_per_l:', action_per_l)

    action_cnt_list = [action_list.count(x) for x in action_day]  # 每天的行为次数
    # print('action_cnt_list:', action_cnt_list)
    action_max = 0 if action_list_len == 0 else max(action_cnt_list)
    # print('action_max:', action_max)

    action_sd = 0 if len(action_day) <= 1 else np.std(action_cnt_list, ddof=1)
    # print('action_sd:', action_sd)

    page_list = [x[2] for x in a]
    page_cnt_list = np.array([page_list.count(x)
                              for x in range(5)])  # 行为在5个页面中发生的数目
    # print('page_cnt_list:', page_cnt_list)
    page_per_d = page_cnt_list / register_time
    # print('page_per_d:', page_per_d)
    page_per_l = page_cnt_list / launch_time
    # print('page_per_l:', page_per_l)
    page_sd = np.std(page_cnt_list, ddof=1)
    # print('page_sd:', page_sd)

    type_list = [x[5] for x in a]
    type_cnt_list = np.array([type_list.count(x) for x in range(6)])
    # print('type_cnt_list:', type_cnt_list)
    type_per_d = type_cnt_list / register_time
    # print('type_per_d:', type_per_d)
    type_per_l = type_cnt_list / launch_time
    # print('type_per_l:', type_per_l)
    type_sd = np.std(type_cnt_list, ddof=1)
    # print('type_sd:', type_sd)

    video_list = [x[3] for x in a]
    video_deduplicate = set(video_list)  # 去重后的video_id
    video_num = len(video_deduplicate)  # 所有行为涉及的视频的个数
    # print('video_num:', video_num)
    video_num_ratio = 0 if action_list_len == 0 else video_num / action_list_len
    # print('video_num_ratio:', video_num_ratio)
    video_cnt_list = np.array([video_list.count(x) for x in video_deduplicate])
    # print('video_cnt_list:', video_cnt_list)
    video_sd = 0 if video_num <= 1 else np.std(video_cnt_list, ddof=1)
    # print('video_sd:', video_sd)

    author_list = [x[4] for x in a]
    author_deduplicate = set(author_list)  # 去重后的author_id
    author_num = len(author_deduplicate)  # 所有行为涉及的视频的个数
    # print('author_num:', author_num)
    author_num_ratio = 0 if action_list_len == 0 else author_num / action_list_len
    # print('author_num_ratio:', author_num_ratio)
    author_cnt_list = np.array([author_list.count(x)
                                for x in author_deduplicate])
    # print('author_cnt_list:', author_cnt_list)
    author_sd = 0 if author_num <= 1 else np.std(author_cnt_list, ddof=1)
    # print('author_sd:', author_sd)

    video_acted_avg = 0 if create_num == 0 else author_id.count(
        user_id) / create_num
    # print('video_acted_avg:', video_acted_avg)

    return np.array([
        register_time,
        register_type,
        device_type,
        active_ratio,
        no_l_ratio,
        l_last_ratio,
        active_2_ratio,
        active_4_ratio,
        active_8_ratio,
        create_num,
        create_per_d,
        create_per_l,
        create_max,
        create_sd,
        create_last,
        create_final_ratio,
        action_per_d,
        action_per_l,
        action_max,
        action_sd
    ] + page_per_d.tolist() + page_per_l.tolist() + [
        page_sd
    ] + type_per_d.tolist() + type_per_l.tolist() + [
        type_sd,
        video_num_ratio,
        video_sd,
        author_num_ratio,
        author_sd,
        video_acted_avg
    ]
    )


def consecutive_days(l: list):
    """

    :param l:
    # :param model: 1为连续非零，2为连续为零
    # :param tol: 所容忍的间隔一天的次数
    :return: 最大连续数
    """

    # 注意，默认输入的l是已经经过从小到大排序的

    # 列表为空时，返回0
    if len(l) == 0:
        return 0

    cons = np.ones(len(l), dtype=np.int64)
    for i in range(len(l) - 1):
        if l[i + 1] == (l[i] + 1):
            cons[i + 1] = cons[i] + 1
    return cons.max()


if __name__ == '__main__':
    test_user_index = [1, 2, 3, 4, 5, 6, 7, 8]
    r, l, c, a = load_data()
    result = []
    author_id = list(a['author_id'].get_values())
    for index in test_user_index:
        r_temp = r.loc[index].get_values()
        l_temp = l[l.user_id == r.loc[index]['user_id']].get_values()
        c_temp = c[c.user_id == r.loc[index]['user_id']].get_values()
        a_temp = a[a.user_id == r.loc[index]['user_id']].get_values()

        result.append(vec(r_temp, l_temp, c_temp, a_temp, author_id, 24))

    print('result.shape:', result[0].shape)
