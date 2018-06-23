# -*- coding: utf-8 -*-
# Created on Tue Jun 19 11:29:28 2018


"""
这是阿里巴巴广告算法打算的一个model的baseline版本
"""

# 加载需要用到的模块
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载训练集
df = pd.read_csv(r'C:\Users\Administrator\Desktop\ali_ad\OfflineTrain.csv')

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
for _ in range(20):
    x_train = pd.concat([x_train, x_train_1])
    y_train = pd.concat([y_train, y_train_1])
    
    x_test = pd.concat([x_test, x_test_1])
    y_test = pd.concat([y_test, y_test_1])

X = np.c_[x_train]
y = np.c_[y_train]


 # 使用逻辑回归模型
 # 定义sigmoid函数
def sigmoid(z):

    return(1 / (1 + np.exp(-z)))
    
#定义损失函数
def costFunction(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + + (reg/(2.0*m))*np.sum(np.square(theta[1:]))
               
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

#求解梯度
def gradient(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1))) 
    
    grad =(1.0/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]

    return(grad.flatten())

# 定义预测函数
def predict(X, threshold =0.5):
    p = sigmoid(X.dot(res.x.T)) # 预测概率
    clf =(p >= threshold)       # 以0.5为基准判断是否交易
    return(p, clf.astype('int'))

reg = 10 # 正则项惩罚系数
initial_theta = np.zeros(X.shape[1])

cost = costFunction(initial_theta, reg, X, y)

grad = gradient(initial_theta, reg, X, y)

print('\n=============模型训练开始============\n')
print('初始化的Cost: \n', cost)
print('初始化的Grad: \n', grad)

# 最小化损失函数
res = minimize(costFunction, initial_theta, args=(reg, X, y), jac=gradient, options={'maxiter':400})

print('\n=============模型训练完成============\n')
print('最终收敛的Cost: \n', res.fun)
print('最终稳定的Grad: \n', res.jac.tolist())

# 对划分的验证集的交易进行预测
y_pred = predict(x_test)[1] 

# 评价模型标准
score = accuracy_score(y_test, y_pred) # 准确率
score_matrix = confusion_matrix(y_test, y_pred) # 结果混淆矩阵

print('该模型在验证集的预测准确率是：{:.2f}'.format(score))
print('该模型在验证集预测结果混淆矩阵是：\n{}'.format(score_matrix))

print('\n\n=============用模型对测试集预测============\n\n')

# 加载测试集
df_test = pd.read_csv(r'C:\Users\Administrator\Desktop\ali_ad\OfflineTest.csv')

# 选取特征列并进行预测
data_x = df_test[feature_cols]
test_pred_list = predict(data_x)[0].tolist()  # 预测的概率列表
test_pred_count = predict(data_x)[1].tolist() # 预测的二分类列表

# 购买概率
p = np.count_nonzero(test_pred_list) / len(test_pred_list)

# 对数损失函数log-loss的计算
ll = 0
for i, p in zip(test_pred_count, test_pred_list):
    ll += -(i * math.log(p) + (1 - i) * math.log(1 - p))
ll = ll / len(test_pred_list) # 对计算结果取均值

print('测试集总人数是:{}\n该模型的损失函数是:{:.2f}'.format(len(test_pred_list),ll))
print('预测会交易的人数将是:{}'.format(np.count_nonzero(test_pred_count)))