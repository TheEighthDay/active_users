import pandas as pd

def generate_label(user_register, app_launch):
    test_data = app_launch[app_launch['day']>=24]
    positive_user = test_data[['user_id']].drop_duplicates()
    positive_user['flag'] = 1
    label = pd.merge(positive_user, user_register[['user_id']], on='user_id', how='outer').fillna(0)
    #去除最近7天内注册的用户
    user_register_7_ago = user_register[user_register['register_day'] < 24][['user_id']]
    label = pd.merge(user_register_7_ago, label, on='user_id', how='left')
    return label