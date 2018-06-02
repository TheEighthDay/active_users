import xgboost as xgb
import numpy as np
from sklearn import metrics

def train_xgb():
    train_feature = np.load("./feature_data/train_feature.npy")
    train_label = np.load("./feature_data/train_label.npy")
    test_feature = np.load("./feature_data/test_feature.npy")
    test_label = np.load("./feature_data/test_label.npy")

    train_data = xgb.DMatrix(train_feature, label=train_label)
    test_data = xgb.DMatrix(test_feature)

    num_round = 500
    param = {'max_depth':3, 'eta':0.01, 'silent':0, 'objective':'binary:logistic', 'lambda': 0.3, 'alpha':0.2, 'eval_metric':'auc'}

    bst = xgb.train(param, train_data, num_round)
    bst.save_model("./saved_model/xgb_model")

    preds = bst.predict(test_data)
    predict_label = []
    for i in preds:
        if i>0.5:
            predict_label.append(1)
        else:
            predict_label.append(0)

    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(test_label.shape[0]):
        if test_label[i] == 1 and predict_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and predict_label[i] == 0:
            FN += 1
        elif test_label[i] == 0 and predict_label[i] == 1:
            FP += 1
        else:
            TN += 1

    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    acc = (TP + TN)/(TP + TN + FN + FP)
    f1_score = 2*precision*recall/(precision+recall)
    print("*"*20)
    print("recall is %f"%(recall))
    print("precision is %f"%(precision))
    print("f1_score is %f"%(f1_score))
    print("acc is %f"%(acc))
    print("*"*20)

    # print(metrics.confusion_matrix(test_feature, predict_label))
    # print("f1_score is %f"%(metrics.f1_score(test_feature, predict_label)))
    # print("acc is %f"%(metrics.precision_score(test_feature, predict_label)))
    # print("recall is %f"%(metrics.recall_score(test_feature, predict_label)))

def predict_xgb(user_feature, user_list):

    predict_feature = xgb.DMatrix(user_feature)
    # load model
    bst = xgb.Booster({'nthread': 4})
    bst.load_model(fname="./saved_model/xgb_model")
    predict_prob = bst.predict(predict_feature)

    predict_prob_sort = sorted(predict_prob, reverse=True)
    threshold = predict_prob_sort[33000]
    print("threshold is %f"%(threshold))

    result = []
    for i in range(len(predict_prob)):
        if predict_prob[i] >= threshold:
            result.append(user_list[i])

    with open("./result/submission.txt", 'w', encoding='utf-8') as f:
        for i in result:
            f.write(str(i))
            f.write("\n")