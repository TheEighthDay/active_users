### 思路

以user_id为index设计特征向量，算法未定。  

---

##### 数据划分

滑动窗口，划分出多个训练集，一是为了扩大训练数据，二是为了方便做模型的交叉验证。  
30/7 ≈ 21/5 ≈ 17/4 ≈ 13/3 ≈ 9/2 ≈ 4/1

|特征提取窗口|标签提取窗口|
|:-:|:-:|
|1-4|5|
|2-10|11-12|
|3-15|16-18|
|4-20|21-24|
|5-25|26-30|

---

##### 特征向量

|列名|维度|备注|
|:-:|:-:|:-|
|register_time|1|注册时长：当前时间 - 注册时间|
|register_type|1|原始数据|
|device_type|1|原始数据|
|active_ratio|1|活跃比例：登陆天数 / register_time|
|active_2_ratio|1|半活跃比例：(用户后半注册时间中登录的天数) / int(register_time / 2)|
|active_4_ratio|1|四活跃比例：(用户后四分之一注册时间中登录的天数) / int(register_time / 4)|
|active_8_ratio|1|八活跃比例：(用户后八分之一注册时间中登录的天数) / int(register_time / 8)|
|no_l_ratio|1|未登录时间比例：(当前时间 - 最后一次登录时间) / register_time|
|l_last_ratio|1|持续活跃比例：连续登陆的天数 / register_time|
|create_num|1|拍摄次数|
|create_per_d|1|平均拍摄比例：create_num / register_time|
|create_per_l|1|登陆拍摄比例：create_num / 登陆天数|
|create_max|1|最大拍摄数|
|create_sd|1|拍摄数标准差：SD(注册后每天的拍摄数)|
|create_last|1|持续拍摄比例：连续拍摄的天数 / register_time|
|create_final_ratio|1|最后拍摄的时间比例：(当前时间 - 最后一次拍摄时间) / register_time|
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
|video_acted_avg|1|视频平均被动作次数：(user_id作为author_id出现的次数) / create_num
