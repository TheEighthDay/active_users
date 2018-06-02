import numpy as np
import pandas as pd
import random

def generate_train_test_dataset(user_label, feature_vector_df, valid_split_ratio):
    sample_number = len(user_label)
    test_sample_number = int(sample_number*valid_split_ratio)

    test_line_num_list = []
    for i in range(test_sample_number):
        test_line_num_list.append(random.randint(0, sample_number-1))
    train_line_num_list = [i for i in range(sample_number) if i not in test_line_num_list]

    train_label = user_label.iloc[train_line_num_list]
    test_label = user_label.iloc[test_line_num_list]

    train_feature = pd.merge(train_label[['user_id']], feature_vector_df, on='user_id', how='left')
    test_feature = pd.merge(test_label[['user_id']], feature_vector_df, on='user_id', how='left')

    return train_feature, train_label, test_feature, test_label