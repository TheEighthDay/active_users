import pandas as pd
import numpy as np

def read_data():
    """
    读取数据
    :return: 四个数据dataframe
    """
    app_launch = pd.read_csv("./original_data/app_launch_log.txt", sep='\t', names=['user_id', 'day'])
    user_register = pd.read_csv("./original_data/user_register_log.txt", sep='\t',
                                names=['user_id', 'register_day', 'register_type', 'device_type'])
    video_create = pd.read_csv("./original_data/video_create_log.txt", sep='\t', names=['user_id', 'day'])
    user_activity = pd.read_csv("./original_data/user_activity_log.txt", sep="\t",
                                names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'])

    return app_launch, user_register, video_create, user_activity


def app_lunch_feature(app_launch):
    """
    计算app_launch的统计特征
    :param app_launch: app_launch dataframe
    :return: app launch statistic
    """
    last_day = app_launch.max()['day']
    launch_count_all =  app_launch\
        .groupby("user_id").count()\
        .rename(columns={'day': 'lunch_count_all'}).reset_index()
    launch_count_1 = app_launch[app_launch["day"] == last_day]\
        .groupby("user_id").count()\
        .rename(columns={'day': 'lunch_count_1'}).reset_index()
    launch_count_3 = app_launch[app_launch["day"] > last_day-3] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'lunch_count_3'}).reset_index()
    launch_count_5 = app_launch[app_launch["day"] > last_day-5] \
            .groupby("user_id").count() \
            .rename(columns={'day': 'lunch_count_5'}).reset_index()
    launch_count_7 = app_launch[app_launch["day"] > last_day - 7] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'lunch_count_7'}).reset_index()
    launch_count_15 = app_launch[app_launch["day"] > last_day - 15] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'lunch_count_15'}).reset_index()

    launch_statistic = pd.merge(launch_count_all, launch_count_1, on='user_id', how='left')
    launch_statistic = pd.merge(launch_statistic, launch_count_3, on='user_id', how='left')
    launch_statistic = pd.merge(launch_statistic, launch_count_5, on='user_id', how='left')
    launch_statistic = pd.merge(launch_statistic, launch_count_7, on='user_id', how='left')
    launch_statistic = pd.merge(launch_statistic, launch_count_15, on='user_id', how='left')
    launch_statistic = launch_statistic.fillna(0)

    return launch_statistic

def user_register_feature(user_register):
    """
    处理用户注册信息
    :param user_register:user_register dataframe
    :return:
    """
    #对device_type进行映射
    device_type_count = user_register[['user_id', 'device_type']].groupby('device_type')\
        .count().rename(columns={'user_id': 'device_type_count'}).reset_index()
    def device_type_map(x):
        if x < 8:
            return -1
        else:
            return x
    device_type_count['new_device_type'] = device_type_count['device_type_count'].map(lambda x:device_type_map(x))
    #对new_device_type进行one-hot编码
    device_type_list = list(set(list(device_type_count['new_device_type'])))
    device_type_one_hot = pd.get_dummies(device_type_list).T.reset_index().rename(columns={'index':'new_device_type'})
    device_type_count = pd.merge(device_type_count, device_type_one_hot, on='new_device_type', how='left')
    #统计register_type
    register_type_count = user_register[['user_id', 'register_type']].groupby('register_type')\
        .count().rename(columns={'user_id': 'register_type_count'}).reset_index()
    #对register_type进行one-hot编码
    register_type_list = list(set(list(register_type_count['register_type'])))
    register_type_one_hot = pd.get_dummies(register_type_list).T.reset_index().rename(columns={'index':'register_type'})
    register_type_count = pd.merge(register_type_count, register_type_one_hot, on='register_type', how='left')
    #合并
    user_register_feature = pd.merge(user_register, device_type_count, on="device_type", how='left')
    user_register_feature = pd.merge(user_register_feature, register_type_count, on="register_type", how='left')
    del user_register_feature['device_type']

    return user_register_feature

def video_create_feature(video_create):

    last_day = video_create.max()['day']
    video_create_count_all =  video_create\
        .groupby("user_id").count()\
        .rename(columns={'day': 'video_create_count_all'}).reset_index()
    video_create_count_1 = video_create[video_create["day"] == last_day]\
        .groupby("user_id").count()\
        .rename(columns={'day': 'video_create_count_1'}).reset_index()
    video_create_count_3 = video_create[video_create["day"] > last_day-3] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'video_create_count_3'}).reset_index()
    video_create_count_5 = video_create[video_create["day"] > last_day-5] \
            .groupby("user_id").count() \
            .rename(columns={'day': 'video_create_count_5'}).reset_index()
    video_create_count_7 = video_create[video_create["day"] > last_day - 7] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'video_create_count_7'}).reset_index()
    video_create_count_15 = video_create[video_create["day"] > last_day - 15] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'video_create_count_15'}).reset_index()

    video_create_statistic = pd.merge(video_create_count_all, video_create_count_1, on='user_id', how='left')
    video_create_statistic = pd.merge(video_create_statistic, video_create_count_3, on='user_id', how='left')
    video_create_statistic = pd.merge(video_create_statistic, video_create_count_5, on='user_id', how='left')
    video_create_statistic = pd.merge(video_create_statistic, video_create_count_7, on='user_id', how='left')
    video_create_statistic = pd.merge(video_create_statistic, video_create_count_15, on='user_id', how='left')
    video_create_statistic = video_create_statistic.fillna(0)

    return video_create_statistic

def action_feature(action, type):
    """
    每种行为的统计信息
    :param action:
    :return:
    """
    last_day = action.max()['day']
    action_count_all =  action\
        .groupby("user_id").count()\
        .rename(columns={'day': 'action_' + str(type) + '_count_all'}).reset_index()
    action_count_1 = action[action["day"] == last_day]\
        .groupby("user_id").count()\
        .rename(columns={'day': 'action_' + str(type) + '_count_1'}).reset_index()
    action_count_3 = action[action["day"] > last_day-3] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'action_' + str(type) + '_count_3'}).reset_index()
    action_count_5 = action[action["day"] > last_day-5] \
            .groupby("user_id").count() \
            .rename(columns={'day': 'action_' + str(type) + '_count_5'}).reset_index()
    action_count_7 = action[action["day"] > last_day - 7] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'action_' + str(type) + '_count_7'}).reset_index()
    action_count_15 = action[action["day"] > last_day - 15] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'action_' + str(type) + '_count_15'}).reset_index()

    action_statistic = pd.merge(action_count_all, action_count_1, on='user_id', how='left')
    action_statistic = pd.merge(action_statistic, action_count_3, on='user_id', how='left')
    action_statistic = pd.merge(action_statistic, action_count_5, on='user_id', how='left')
    action_statistic = pd.merge(action_statistic, action_count_7, on='user_id', how='left')
    action_statistic = pd.merge(action_statistic, action_count_15, on='user_id', how='left')
    action_statistic = action_statistic.fillna(0)

    return action_statistic


def page_feature(page, type):
    """
    每个页面的统计信息
    :param page:
    :return:
    """
    last_day = page.max()['day']
    page_count_all =  page\
        .groupby("user_id").count()\
        .rename(columns={'day': 'page_' + str(type) + '_count_all'}).reset_index()
    page_count_1 = page[page["day"] == last_day]\
        .groupby("user_id").count()\
        .rename(columns={'day': 'page_' + str(type) + '_count_1'}).reset_index()
    page_count_3 = page[page["day"] > last_day-3] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'page_' + str(type) + '_count_3'}).reset_index()
    page_count_5 = page[page["day"] > last_day-5] \
            .groupby("user_id").count() \
            .rename(columns={'day': 'page_' + str(type) + '_count_5'}).reset_index()
    page_count_7 = page[page["day"] > last_day - 7] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'page_' + str(type) + '_count_7'}).reset_index()
    page_count_15 = page[page["day"] > last_day - 15] \
        .groupby("user_id").count() \
        .rename(columns={'day': 'page_' + str(type) + '_count_15'}).reset_index()

    page_statistic = pd.merge(page_count_all, page_count_1, on='user_id', how='left')
    page_statistic = pd.merge(page_statistic, page_count_3, on='user_id', how='left')
    page_statistic = pd.merge(page_statistic, page_count_5, on='user_id', how='left')
    page_statistic = pd.merge(page_statistic, page_count_7, on='user_id', how='left')
    page_statistic = pd.merge(page_statistic, page_count_15, on='user_id', how='left')
    page_statistic = page_statistic.fillna(0)
    return page_statistic

def user_activity_feature(user_activity):
    #行为
    user_action_0 = user_activity[user_activity['action_type'] == 0][['user_id', 'day']]
    user_action_1 = user_activity[user_activity['action_type'] == 1][['user_id', 'day']]
    user_action_2 = user_activity[user_activity['action_type'] == 2][['user_id', 'day']]
    user_action_3 = user_activity[user_activity['action_type'] == 3][['user_id', 'day']]
    user_action_4 = user_activity[user_activity['action_type'] == 4][['user_id', 'day']]
    user_action_5 = user_activity[user_activity['action_type'] == 5][['user_id', 'day']]

    action_statistic_0 = action_feature(user_action_0, 0)
    action_statistic_1 = action_feature(user_action_1, 1)
    action_statistic_2 = action_feature(user_action_2, 2)
    action_statistic_3 = action_feature(user_action_3, 3)
    action_statistic_4 = action_feature(user_action_4, 4)
    action_statistic_5 = action_feature(user_action_5, 5)
    #页面
    user_page_0 = user_activity[user_activity['page'] == 0][['user_id', 'day']]
    user_page_1 = user_activity[user_activity['page'] == 1][['user_id', 'day']]
    user_page_2 = user_activity[user_activity['page'] == 2][['user_id', 'day']]
    user_page_3 = user_activity[user_activity['page'] == 3][['user_id', 'day']]
    user_page_4 = user_activity[user_activity['page'] == 4][['user_id', 'day']]

    page_statistic_0 = page_feature(user_page_0, 0)
    page_statistic_1 = page_feature(user_page_1, 1)
    page_statistic_2 = page_feature(user_page_2, 2)
    page_statistic_3 = page_feature(user_page_3, 3)
    page_statistic_4 = page_feature(user_page_4, 4)

    user_action_all = pd.merge(user_activity[['user_id']].drop_duplicates(), action_statistic_0, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, action_statistic_1, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, action_statistic_2, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, action_statistic_3, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, action_statistic_4, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, action_statistic_5, on='user_id', how='left')

    user_action_all = pd.merge(user_action_all, page_statistic_0, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, page_statistic_1, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, page_statistic_2, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, page_statistic_3, on='user_id', how='left')
    user_action_all = pd.merge(user_action_all, page_statistic_4, on='user_id', how='left')

    user_action_all.fillna(0)
    return user_action_all
