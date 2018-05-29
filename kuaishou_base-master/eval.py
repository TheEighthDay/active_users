def result_eval(predict_user_id, real_user_id):
    """
    评估函数
    :param predict_user_id: 预测的用户列表
    :param real_user_id: 真实的用户列表
    :return: 返回精准率， 召回率， f1值
    """
    intersection = list(set(real_user_id).intersection(set(predict_user_id)))

    precision = len(intersection)/len(predict_user_id)
    recall = len(intersection)/len(real_user_id)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score