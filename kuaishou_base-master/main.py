from classifier_model import rf_model
import pickle
import pandas as pd
import numpy as np

#train
rf_model.train_rf_cv()

#predict
with open("./feature_data/predict_user_list.pkl", 'rb') as f:
    user_list = pickle.load(f)

user_feature = np.load("./feature_data/predict_feature.npy")
predict_prob = rf_model.predict_rf(user_feature, user_list)

np.save("./result_analysis/predict_prob", predict_prob)
