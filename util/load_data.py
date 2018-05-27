#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : load_data.py
# @Author: JohnHuiWB
# @Date  : 2018/5/26 0026
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

import tqdm
import pandas as pd

def o_data_2_csv_app_launch_log():
    """
    将所有的原始文件写成小于100MB的CSV文件，便于pandas操作
    :return:
    """
    user_id = []
    day = []
    with open('../original_data/app_launch_log.txt', 'r', encoding='utf8') as fp:
        for l in tqdm.tqdm(fp.readlines()):
            l = l.split('\t')
            user_id.append(int(l[0]))
            day.append(int(l[1]))
    col_user_id = pd.Series(user_id, name='user_id')
    col_day = pd.Series(day, name='day')

    # 用concat方式连接后存储，使得列的顺序不变
    con = pd.concat([col_user_id, col_day], axis=1)
    con.to_csv('../original_data/app_launch_log.csv', index=False, sep=' ')


def o_data_2_csv_user_activity_log():
    user_id = []
    day = []
    page = []
    video_id = []
    author_id = []
    action_type = []
    cnt = 0
    num = 1
    with open('../original_data/user_activity_log.txt', 'r', encoding='utf8') as fp:
        for l in tqdm.tqdm(fp.readlines()):
            cnt += 1

            l = l.split('\t')
            user_id.append(int(l[0]))
            day.append(int(l[1]))
            page.append(int(l[2]))
            video_id.append(int(l[3]))
            author_id.append(int(l[4]))
            action_type.append(int(l[5]))

            if cnt == 3400000:
                col_user_id = pd.Series(user_id, name='user_id')
                col_day = pd.Series(day, name='day')
                col_page = pd.Series(page, name='page')
                col_video_id = pd.Series(video_id, name='video_id')
                col_author_id = pd.Series(author_id, name='author_id')
                col_action_type = pd.Series(action_type, name='action_type')
                con = pd.concat([col_user_id, col_day, col_page, col_video_id, col_author_id, col_action_type], axis=1)
                con.to_csv('../original_data/user_activity_log' + str(num) + '.csv', index=False, sep=' ')
                user_id.clear()
                day.clear()
                page.clear()
                video_id.clear()
                author_id.clear()
                action_type.clear()
                num += 1
                cnt = 0

    col_user_id = pd.Series(user_id, name='user_id')
    col_day = pd.Series(day, name='day')
    col_page = pd.Series(page, name='page')
    col_video_id = pd.Series(video_id, name='video_id')
    col_author_id = pd.Series(author_id, name='author_id')
    col_action_type = pd.Series(action_type, name='action_type')
    con = pd.concat([col_user_id, col_day, col_page, col_video_id, col_author_id, col_action_type], axis=1)
    con.to_csv('../original_data/user_activity_log' + str(num) + '.csv', index=False, sep=' ')


def o_data_2_csv_user_register_log():
    user_id = []
    register_day = []
    register_type = []
    device_type = []
    with open('../original_data/user_register_log.txt', 'r', encoding='utf8') as fp:
        for l in tqdm.tqdm(fp.readlines()):
            l = l.split('\t')
            user_id.append(int(l[0]))
            register_day.append(int(l[1]))
            register_type.append(int(l[2]))
            device_type.append(int(l[3]))
    col_user_id = pd.Series(user_id, name='user_id')
    col_register_day = pd.Series(register_day, name='register_day')
    col_register_type = pd.Series(register_type, name='register_type')
    col_device_type = pd.Series(device_type, name='device_type')
    con = pd.concat([col_user_id, col_register_day, col_register_type, col_device_type], axis=1)
    con.to_csv('../original_data/user_register_log.csv', index=False, sep=' ')


def o_data_2_csv_video_create_log():
    user_id = []
    day = []
    with open('../original_data/video_create_log.txt', 'r', encoding='utf8') as fp:
        for l in tqdm.tqdm(fp.readlines()):
            l = l.split('\t')
            user_id.append(int(l[0]))
            day.append(int(l[1]))
    col_user_id = pd.Series(user_id, name='user_id')
    col_day = pd.Series(day, name='day')
    con = pd.concat([col_user_id, col_day], axis=1)
    con.to_csv('../original_data/video_create_log.csv', index=False, sep=' ')


def load_data():
    """

    :return: r, l, c, a
    """
    app_launch_log = pd.DataFrame(pd.read_csv('../original_data/app_launch_log.csv', sep=' '))
    user_activity_log1 = pd.read_csv('../original_data/user_activity_log1.csv', sep=' ')
    user_activity_log2 = pd.read_csv('../original_data/user_activity_log2.csv', sep=' ')
    user_activity_log3 = pd.read_csv('../original_data/user_activity_log3.csv', sep=' ')
    user_activity_log4 = pd.read_csv('../original_data/user_activity_log4.csv', sep=' ')
    user_activity_log5 = pd.read_csv('../original_data/user_activity_log5.csv', sep=' ')
    user_activity_log6 = pd.read_csv('../original_data/user_activity_log6.csv', sep=' ')
    user_activity_log = pd.DataFrame(pd.concat([user_activity_log1,
                                  user_activity_log2,
                                  user_activity_log3,
                                  user_activity_log4,
                                  user_activity_log5,
                                  user_activity_log6]))
    user_register_log = pd.DataFrame(pd.read_csv('../original_data/user_register_log.csv', sep=' '))
    video_create_log = pd.DataFrame(pd.read_csv('../original_data/video_create_log.csv', sep=' '))

    return user_register_log, app_launch_log, video_create_log, user_activity_log


if __name__ == '__main__':
    o_data_2_csv_app_launch_log()
    o_data_2_csv_user_activity_log()
    o_data_2_csv_user_register_log()
    o_data_2_csv_video_create_log()

    r, l, c, a = load_data()
    print(l.keys())
    print(a.keys())
    print(r.keys())
    print(c.keys())

    pass