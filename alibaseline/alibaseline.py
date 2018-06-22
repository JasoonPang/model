# -*- coding: utf-8 -*-
# Created on Tue Jun 19 11:29:28 2018


"""
这是阿里巴巴广告算法打算的一个model的baseline版本，通过训练，该模型准确率接近0.6
"""

# 加载需要用到的模块
import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载训练集
df = pd.read_csv(r'OfflineTrain.csv')

# 根据是否交易划分数据集
df_0 = df.loc[df['is_trade'] == 0] # 未交易
df_1 = df.loc[df['is_trade'] == 1] # 交易

# 将要选用的特征列
feature_cols = ['item_price_level', 
                 'item_collected_level',
                 'user_gender_id', 
                 'user_age_level', 'shop_star_level',
                 'shop_score_service',
                 'shop_score_delivery',
                 'shop_score_description',
                 ]

# 选取未交易的特征列和标签列
data_x_0 = df_0[feature_cols]
data_y_0 = df_0['is_trade']

# 选取交易的特征列和标签列
data_x_1 = df_1[feature_cols]
data_y_1 = df_1['is_trade']

# 分别从未交易数据集和交易数据集分割出训练集和验证集
x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_x_0, data_y_0, test_size = 0.2) # 未交易数据集
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_x_1, data_y_1, test_size = 0.2) # 交易数据集

# 初始化训练集和验证集
x_train = pd.concat([x_train_0, x_train_1])
y_train = pd.concat([y_train_0, y_train_1])
x_test = pd.concat([x_test_0, x_test_1])
y_test = pd.concat([y_test_0, y_test_1])

# 对交易样本进行过采样
for _ in range(50):
    x_train = pd.concat([x_train, x_train_1])
    y_train = pd.concat([y_train, y_train_1])
    
    x_test = pd.concat([x_test, x_test_1])
    y_test = pd.concat([y_test, y_test_1])


 # 使用逻辑回归模型
clf = LogisticRegressionCV()

clf.fit(x_train, y_train)  # 拟合数据
y_pred = clf.predict(x_test) # 预测


# 评价模型标准
score = accuracy_score(y_test, y_pred)
score_matrix = confusion_matrix(y_test, y_pred)

print('该模型在验证集的预测准确率是：{:.2f}'.format(score))
print('该模型在验证集预测结果混淆矩阵是：{}'.format(score_matrix))

# 加载测试集
df_test = pd.read_csv(r'OfflineTest.csv')

# 选取特征列并进行预测
data_x = df_test[feature_cols]
y_test_pred = clf.predict(data_x)

# 购买概率
p = np.count_nonzero(y_test_pred) / len(y_test_pred)

# 对数损失函数
ll = -p * math.log(p)

print('测试集总人数是:{}\n该模型的损失函数是:{:.2f}'.format(len(y_test_pred),ll))
print('预测会交易的人数将是:{}'.format(np.count_nonzero(y_test_pred)))