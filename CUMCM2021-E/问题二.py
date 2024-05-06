import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data0 = pd.read_csv(r'附件2.csv', index_col = 0)
data0.shape

data0.head()

data0.describe([0.01, 0.25,0.75, 0.99]).T

data0.info()

data0.iloc[:,1:].isnull().any().any()

data0.iloc[:,0].isnull().sum()

data0.iloc[:,0].value_counts().sort_index()

data_X = data0.iloc[:,1:]
print(data_X.shape)
data_X = pd.DataFrame(data_X.iloc[:,1:].values- data_X.iloc[:,:-1].values, index = data0.index)
print(data_X.shape)
print(data_X)
data_y = data0.iloc[:,0]
print(data_X.shape, data_y.shape)
print(f'缺失值个数:{data_y.isnull().sum()}个')
data_y.fillna(0, inplace = True)  #暂时用0填充缺失值
print(f'缺失值个数:{(data_y == 0).sum()}个')
#转换为整数
data_y = data_y.astype('int')  #保留缺失值的索引号
qsz_index = data_y[data_y == 0].index
qsz_index

def my_plot(x):
    plt.plot(x.index, x.values, linewidth =0.5)

def PlotSpectrum(data, strO):
    fontsize = 5
    plt.figure(strO, figsize= (5, 3), dpi = 300)
    plt.xticks(range(0, 4001, 500),rotation = 45, fontsize = 5)
    plt.yticks( fontsize = fontsize)
    plt.xlabel('波段', fontsize = 6)
    plt.ylabel('吸光度', fontsize = 6)
    plt.grid(True)
    data.agg(lambda x: my_plot(x), axis = 1)
    plt.show()

PlotSpectrum(data0.iloc[:,1:], '附件二光谱数据曲线图')

PlotSpectrum(data_X, '附件二一阶平滑后数据曲线图')

max_min = data_X.agg(lambda x: x.max() - x.min())
print(max_min.describe())
data1 = data_X.loc[:,max_min > max_min.mean()+ 0* max_min.std()]
# data1 = data_X
print(data_X.shape, data1.shape)

data_X_know = data1.drop(qsz_index)
data_X_unknow = data1.loc[qsz_index,:]
print(data_X_know.index, 'in', data_X_unknow.index)
data_y_know = data_y.drop(qsz_index)
print(data_X_know.shape, data_y_know.shape, data_X_unknow.shape)
test = pd.concat([data_y_know, data_X_know],axis = 1)
test.shape

from scipy.spatial import distance_matrix
dis = pd.DataFrame(distance_matrix(data_X_know.values, data_X_unknow.values), index =data_X_know.index, columns = data_X_unknow.index)
dis_group = pd.concat([data_y_know, dis], axis = 1)
print(dis_group.shape)
print(dis_group.columns)
print(dis_group)

min_index = dis_group.idxmin()
print(min_index)
[*zip(qsz_index, dis_group.loc[min_index[1:], 'OP'])]

dis_groupby = dis_group.groupby('OP').mean()
dis_groupby

by_min_index = dis_groupby.idxmin()
print(by_min_index)

dis = pd.DataFrame(distance_matrix(data_X_know.values, data_X_know.values), index =data_X_know.index, columns = data_X_know.index)
dis_group = pd.concat([data_y_know, dis], axis = 1)
print(dis_group.shape)
print(dis_group.columns)
print(dis_group)

OP_index = [0 for i in range(12)]
for i in range(1, 12):
    OP_index[i] = dis_group[dis_group['OP'] == i].index
OP_index

dis=pd.DataFrame(distance_matrix(data_X_know.loc[OP_index[1],:].values,data_X_know.loc[OP_index[1],:].values)).mean().mean()
print(dis)
dis=pd.DataFrame(distance_matrix(data_X_know.loc[OP_index[1],:].values,data_X_know.loc[OP_index[2],:].values)).mean().mean()
dis

dis_g = pd.DataFrame(np.zeros([11,11]), index = range(1, 12), columns = range(1, 12))
for i in range(1, 12):
    for j in range(1,12):
        dis_g.loc[i, j] =pd.DataFrame(distance_matrix(data_X_know.loc[OP_index[i],:].values,data_X_know.loc[OP_index[i],:].values)).mean().mean()
dis_g

dis_g.idxmin()

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print(data_X.shape)
best_LLE_clf_n_components,best_LLE_clf_n_neighbors =35, 70
best_params_ ={'C':2.438775510204082, 'gamma': 18.420699693267164}
print(f'n_components={best_LLE_clf_n_components},n_neighbors={best_LLE_clf_n_neighbors}')
# print(f'best_params_ = igrid.best_params_}')
lle = LocallyLinearEmbedding(n_components = best_LLE_clf_n_components,n_neighbors = best_LLE_clf_n_neighbors, method = 'modified').fit(data1) # LLE降维

data_X_lle = lle.transform(data1)
#获取降维后的数据
data_X_lle = pd.DataFrame(data_X_lle)
#转换为DataFrame
data_y= pd.Series(data_y)
#转换为Series

#将未分类数据剥离
data_X_lle_know = data_X_lle.drop(qsz_index, axis =0)
#删除未分类的行
data_X_lle_unknow = data_X_lle.loc[qsz_index,:]
#取出未分类的行
print(data_X_lle_know.shape, data_X_lle_unknow.shape, data_y_know.shape)
# clf = svc(C = best_params_I'C], kernel = 'rbf，gamma = best_params_['gamma'],decision_function_shape = 'ovr', cache_size = 5000 
# cvs = cross_val_score(clf, data_X_lle_know, data_y_know, cVv= 10)
# print(f'预测平均准确率:{cvs.mean()1")
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_X_lle_know, data_y_know,test_size = 0.3)
#分离出测试集和训练集
clf = SVC(C = best_params_['C'], kernel = 'rbf' , gamma = best_params_['gamma'],decision_function_shape = 'ovr', cache_size =5000). fit(Xtrain, Ytrain)
score_r = clf.score(Xtest, Ytest)
print(score_r)

clf = clf.fit(data_X_lle_know, data_y_know)
print(clf.score(data_X_lle_know, data_y_know))
print(clf.predict(data_X_lle_unknow))

