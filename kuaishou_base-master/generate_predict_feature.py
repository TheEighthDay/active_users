import pickle
import pandas as pd
import numpy as np

from data_process import feature_project

app_launch, user_register, video_create, user_activity = feature_project.read_data()
launch_statistic = feature_project.app_lunch_feature(app_launch)
video_create_statistic = feature_project.video_create_feature(video_create)
user_register_statistic = feature_project.user_register_feature(user_register)
user_activity_statistic = feature_project.user_activity_feature(user_activity)

user_feature = pd.merge(user_register_statistic, video_create_statistic, on='user_id', how='left')
user_feature = pd.merge(user_feature, launch_statistic, on='user_id', how='left')
user_feature = pd.merge(user_feature, user_activity_statistic, on='user_id', how='left')

user_feature = user_feature.fillna(0)

user_feature.to_csv("./feature_data/user_feature.csv")
user_list = list(user_feature['user_id'])
del user_feature['user_id']

np.save("./feature_data/predict_feature",user_feature.values)

with open("./feature_data/predict_user_list.pkl", 'wb') as f:
    pickle.dump(user_list, f)