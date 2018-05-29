# kuaishou_base
快手大数据竞赛base_line

### 目录结构
三个package：
 - classifier_model: 构建分类算法，当前主要有xgboost和random forest
 - data_process： 数据处理和特征工程
 - data_analysis: 使用jupyter对日志做基本的数据分析
 
由于空目录不能上传GitHub，需要新建5个目录：
 - original_data: 存放竞赛的4个日志文件
 - feature_data: 用于保存特征向量
 - result_analysis: 用于保存预测的概率
 - saved_model: 用于保存训练好的模型
 - result: 用于保存submission文件
 
### 函数说明：
 - eval.py: 评估函数
 - generate_feature_and_label.py： 生成特征向量和标签，划分训练集和测试集
 - generate_predict_feature.py： 生成预测使用的特征向量
 - main.py: 主函数，训练模型和预测标签
 
 ### 使用说明：
 
 特征工程
 ```
 python generate_feature_and_label.py
 python generate_predict_feature.py
 ```
 训练和预测
 ```
 python main.py
 ```
### 结果：
使用random forest的提交成绩为0.75左右
