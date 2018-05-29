from data_process import feature_project
from data_process import generate_label
from data_process import train_test_data_split

import pandas as pd
import numpy as np

app_launch, user_register, video_create, user_activity = feature_project.read_data()

launch_statistic = feature_project.app_lunch_feature(app_launch[app_launch['day']<24])
video_create_statistic = feature_project.video_create_feature(video_create[video_create['day']<24])
user_register_statistic = feature_project.user_register_feature(user_register)
user_activity_statistic = feature_project.user_activity_feature(user_activity[user_activity['day']<24])

user_feature = pd.merge(user_register_statistic, video_create_statistic, on='user_id', how='left')
user_feature = pd.merge(user_feature, launch_statistic, on='user_id', how='left')
user_feature = pd.merge(user_feature, user_activity_statistic, on='user_id', how='left')

user_feature = user_feature.fillna(0)
user_feature.to_csv("./feature_data/train_user_feature.csv")
user_label = generate_label.generate_label(user_register, app_launch)
train_feature, train_label, test_feature, test_label = train_test_data_split.generate_train_test_dataset(
    user_label=user_label,
    feature_vector_df=user_feature,
    valid_split_ratio=0.2
)

train_feature = train_feature.fillna(0)
test_feature = test_feature.fillna(0)

print(train_feature.isnull().any().any())
print(test_feature.isnull().any().any())

del train_feature['user_id']
del train_label['user_id']
del test_feature['user_id']
del test_label['user_id']

np.save("./feature_data/train_feature", train_feature.values)
np.save("./feature_data/train_label", train_label.values)
np.save("./feature_data/test_feature", test_feature.values)
np.save("./feature_data/test_label", test_label.values)
