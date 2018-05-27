### 思路

以user_id为index设计特征向量，算法未定。  
将30天的数据分为前23天和后7天，前者作为训练集，后者作为测试集。这个划分用于测试所设计的算法的性能。  
最终，将所有数据作为输入进行训练。

---

##### 特征向量

|列名|维度|备注|
|:-:|:-:|:-|
|register_time|1|注册时长：当前时间 - 注册时间|
|register_type|1|原始数据|
|device_type|1|原始数据|
|active_ratio|1|活跃比例：登陆天数 / register_time|
|no_l_ratio|1|未登录时间比例：(当前时间 - 最后一次登录时间) / register_time|
|l_last_ratio|1|持续活跃比例：连续登陆的天数 / register_time|
|highfrequency_bool|1|规律登陆且周期小于等于七天：bool 上下两次登录时间差之差在二以内且登陆周期小于等于七天|
|create_per_d|1|平均拍摄比例：拍摄次数 / register_time|
|create_per_l|1|登陆拍摄比例：拍摄次数 / 登陆天数|
|create_max|1|最大拍摄数
|create_sd|1|拍摄数标准差：SD(注册后每天的拍摄数)|
|create_last|1|持续拍摄比例：连续拍摄的天数 / register_time|
|action_per_d|1|平均行为比例：行为次数 / register_time|
|action_per_l|1|登陆行为比例：行为次数 / 登陆天数|
|action_max|1|最大行为数|
|action_sd|1|行为数标准差：SD(注册后每天的行为数)|
|page_per_d|5|行为在5个页面中发生的数目 / register_time|
|page_per_l|5|行为在5个页面中发生的数目 / 登陆天数|
|page_sd|1|SD(行为在5个页面中发生的数目)|
|type_per_d|6|6个行为类别的数目 / register_time|
|type_per_l|6|6个行为类别的数目 / 登陆天数|
|type_sd|1|SD(6个行为类别的数目)|
|video_num_ratio|1|视频数目比例：所有行为涉及的视频的个数 / 行为次数|
|video_sd|1|SD(对每个视频所做行为的次数)|
|author_num_ratio|1|作者数目比例：所有行为涉及的作者的个数 / 行为次数|
|author_sd|1|SD(对每个作者的视频所做行为的次数)|
|author_bool|1|(bool这个用户是作者（作者每天上线看自己视频情况高）)|
|author_time|1|(最后发表作品时间)|
