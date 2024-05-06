# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:43:34 2024

@author: admin
"""

import numpy as np
import random as rd
# 添加高斯噪音
def add_gaussian_noise(data, mean = 0, std = 0.5):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

# He初始化函数
def initialize_parameters_he(layers_dims):
	"""
	Arguments:
	layer_dims -- python array (list) containing the size of each layer.
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
					b1 -- bias vector of shape (layers_dims[1], 1)
					...
					WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
					bL -- bias vector of shape (layers_dims[L], 1)
	"""
	np.random.seed(3)
	parameters = {}
	L = len(layers_dims)  # integer representing the number of layers
 
	for l in range(1, L):
		parameters['w' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
	return parameters


# 定义神经网络初始参数，进行2分类
input_size = 2
hidden_size = [4, 4]
output_size = 2

data_size = 100
train_percent = 0.8

# 创建数据集
data_x = np.array([i / 10 for i in range(10,data_size + 10)])
data_y = np.array(add_gaussian_noise(np.array([(i / 10) ** 2 for i in range(10,data_size + 10)])))
data = np.hstack((data_x[:, np.newaxis], data_y[:, np.newaxis]))
data_label = [-1] * data_size

for i in range(len(data_y)):
    if data_x[i] ** 2 > data_y[i]:
        data_label[i] = 1
    else:
        data_label[i] = 0

'''
sum = 0
for i in data_label:
    sum+=i'''
data_train ,data_test= [], []
for i in range(int(train_percent * data_size)):
    data_train.append(rd.choice(data))
for i in range(data_size - int(train_percent * data_size)):
    data_test.append(rd.choice(data))
# 初始化权重
weight = initialize_parameters_he([input_size,hidden_size[0],hidden_size[1],output_size])

# 定义前向传播方法
def forward(weight, data):
    data_out = [0] * output_size
    data_hidden = [[0 for j in range(4)] for i in range(2)]
    
    for i in range(np.shape(weight['w1'])[0]):
        data_hidden[0][i] += weight['b1'][i][0]               #偏置 
        for j in range(np.shape(weight['w1'])[1]):
            data_hidden[0][i] += data[j] * weight['w1'][i][j] #权重
            
    for i in range(np.shape(weight['w2'])[0]):
        data_hidden[1][i] += weight['b2'][i][0]                #偏置 
        for j in range(np.shape(weight['w2'])[1]):
            data_hidden[1][i] += data_hidden[0][j] * weight['w2'][i][j] #权重
            
    for i in range(np.shape(weight['w3'])[0]):
        data_out[i] += weight['b3'][i][0]                #偏置 
        for j in range(np.shape(weight['w3'])[1]):
            data_out[i] += data_hidden[1][j] * weight['w3'][i][j] #权重
            
    return data_out









