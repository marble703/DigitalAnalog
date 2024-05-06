import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] =['SimHei']
plt.rcParams['axes.unicode_minus']=False

data0=pd.read_csv(r'附件4.csv',index_col=0)
data0.shape

data0.head()

print(data0['OP'].isnull().any(), data0['Class'].isnull().any())

data0.describe([0.01,0.25, 0.75, 0.99]).T

data0.info()

data0.iloc[:,2:].isnull().any().any()

data0.iloc[:,0].value_counts().sort_index()

data0.iloc[:,1].value_counts().sort_index()

print(data0['OP'].isnull().sum(), data0['Class'].isnull().sum())
OP_null_index = data0[data0['OP'].isnull()].index
Class_null_index = data0[data0['Class'].isnull()].index
print(OP_null_index, 'kn', Class_null_index)
public_index = []
for i in OP_null_index:
    if i in Class_null_index:
        public_index.append(i)
public_index

def my_plot(x):
    plt.plot(x.index, x.values, linewidth = 0.5)

def PlotSpectrum(data, str0):
    fontsize = 5
    plt.figure(str0, figsize = (5, 3), dpi = 300)
    plt.xticks(range(0,8001,500), rotation = 45, fontsize = 5)
    plt.yticks(fontsize = fontsize)
    plt.xlabel('波段', fontsize = 6)
    plt.ylabel('吸光度', fontsize = 6)
    plt.grid(True)
    data.agg(lambda x: my_plot(x), axis = 1)
    plt.show()

PlotSpectrum(data0.iloc[:,2:], '附件四光谱数据曲线图')

data0['Class'].fillna('D' , inplace = True)
data0['OP'].fillna(0, inplace = True)
data0

colors = {'A': 'r', 'B': 'g', 'C': 'b', 'D':'y'}
plt.figure('附件四 Class', figsize = (5, 3), dpi = 300)
plt.xticks(range(0,10000,1000),rotation = 45, fontsize = 5)
plt.yticks(fontsize = 5)
plt.xlabel('波段', fontsize = 5)
plt.ylabel('吸光度', fontsize = 5)
for i in range(data0.shape[0]):
    if data0.iloc[i,0] != 'D':
        plt. plot(data0.columns[2:], data0.iloc[i,2:], c = colors[data0.iloc[i,0]], linewidth = 0.5)
plt.grid(True)
plt.show()

dataBCD = data0[data0['Class'].isin(['B', 'C', 'D'])]
BC_index = dataBCD[dataBCD['Class'].isin(['B', 'C'])].index


data_X = dataBCD.iloc[:,2:]
data_Class = dataBCD.iloc[:,0]
max_min = data_X.agg(lambda x: x.max() - x.min())
print(max_min.describe())
data1 = data_X.loc[:,max_min > max_min.mean()+ 0 * max_min.std()]
print(data_X.shape, data1.shape)
data1

data_Class_know = data1.drop(BC_index)
data_Class_unknow = data1.loc[BC_index,:]
print(data_Class_know.index, '\n', data_Class_unknow.index)

from scipy.spatial import distance_matrix

dis_C = pd.DataFrame(distance_matrix(data_Class_know.values, data_Class_unknow.values),index = data_Class_know.index, columns = data_Class_unknow.index)
dis_Cgroup = pd.concat([data_Class, dis_C], axis = 1)
print(dis_Cgroup.shape)
print(dis_Cgroup.columns)
print(dis_Cgroup)

min_Cindex = dis_Cgroup.iloc[:,1:].idxmin()
print(min_Cindex)
[*zip(BC_index, dis_Cgroup.loc[min_Cindex[1:], 'Class'])]

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print(dataBCD.shape)
best_LLE_clf_n_components, best_LLE_clf_n_neighbors = 20, 40
best_params_= {'C':2.438775510204082, 'gamma': 18.420699693267164}
print(f'n_components{best_LLE_clf_n_components},n_neighbors={best_LLE_clf_n_neighbors}')
# print(f'best_params_ = {grid.best_params_y')
lle = LocallyLinearEmbedding(n_components = best_LLE_clf_n_components, n_neighbors = best_LLE_clf_n_neighbors, method = 'modified' ).fit(dataBCD.iloc[:,2:])#LLE降维
dataBCD_lle = lle.transform(dataBCD.iloc[:,2:])
#获取降维后的数据
dataBCD_lle = pd.DataFrame(dataBCD_lle, index = dataBCD.index)
#转换为 DataFrame
print(dataBCD_lle)

X_know = dataBCD_lle[(dataBCD['Class'] == 'B')| (dataBCD[ 'Class'] == 'C')]
X_unknow = dataBCD_lle[(dataBCD[ 'Class'] == 'D')]
y_know = dataBCD.loc[(dataBCD['Class'] == 'B')|(dataBCD['Class'] == 'C'),'Class']
print(X_know.shape, X_unknow.shape, y_know.shape)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_know, y_know, test_size = 0.3)
#分离出测试集和训练集
clf = SVC(C = best_params_['C'], kernel = 'rbf', gamma = best_params_['gamma'],decision_function_shape = 'ovr', cache_size = 5000).fit(Xtrain, Ytrain) # type: ignore
score_r= clf.score(Xtest,Ytest)
print(score_r)
clf = clf.fit(X_know,y_know)
print(clf.score(X_know, y_know))
print(clf.predict(X_unknow))
resBC = [*zip(BC_index, clf.predict(X_unknow))]

resB, resC = [], []
for i in resBC:
    if i[1]== 'B':
        resB.append(i[0])
    else:
        resC.append(i[0])

print(resBC)
print(resB, len(resB))
print(resC, len(resC))

colors = ["r", "g", "b", "c" , "m" , "y", "navy", "purple", "gold", "gray", "orange", "lime","pink","tomato", "khaki" , "azure" , "linen", "rose", ]
plt.figure('附件四 Class', figsize = (5, 3), dpi = 300)
plt.xticks(range(0, 10000, 1000), rotation = 45, fontsize = 5)
plt.yticks(fontsize = 5)
plt.xlabel('波段', fontsize = 5)
plt.ylabel('吸光度', fontsize = 5)
plt.grid(True)
for i in range(data0.shape[0]):
    plt.plot(data0.columns[2:], data0.iloc[i,2:], color = colors[int(data0.iloc[i,1])], linewidth = 0.5)
plt.show()

best_LLE_clf_n_components, best_LLE_clf_n_neighbors = 20, 40
best_params_={'C': 2.438775510204082, 'gamma': 18.420699693267164}
print(f'n_components={best_LLE_clf_n_components},n_neighbors={best_LLE_clf_n_neighbors}')
# print(f'best_params_ = {grid.best_params_}y)
lle = LocallyLinearEmbedding(n_components = best_LLE_clf_n_components, n_neighbors = best_LLE_clf_n_neighbors, method = 'modified').fit(dataBCD. iloc[:,2:])#LLE降维
dataBCD_lle = lle.transform(dataBCD.iloc[:,2:])
#获取降维后的数据
dataBCD_lle = pd.DataFrame(dataBCD_lle, index = dataBCD.index)
#转换为DataFrame
print(dataBCD_lle)

X_know = dataBCD_lle[(dataBCD['Class'] == 'B')|(dataBCD['Class'] == 'C')]
X_unknow = dataBCD_lle[(dataBCD['Class'] == 'D')]
y_know = dataBCD.loc[(dataBCD['Class'] == 'B')| (dataBCD['Class'] == 'C'),'Class']
print(X_know.shape, X_unknow.shape, y_know.shape)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_know,y_know, test_size = 0.3)
#分离出测试集和训练集
clf = SVC(C = best_params_['C'], kernel = 'rbf',gamma = best_params_['gamma'],decision_function_shape = 'ovr' , cache_size = 5000 ).fit(Xtrain, Ytrain)

score_r = clf.score(Xtest,Ytest)
print(score_r)
clf = clf.fit(X_know,y_know)
print(clf.score(X_know, y_know))
print(clf.predict(X_unknow))
[*zip(BC_index, clf.predict(X_unknow))]

data0['OP'] = data0['OP'].astype(int)
data0

data_X= pd.DataFrame(data0.iloc[:,3:].values - data0.iloc[:,2:-1].values, index = data0.index)
data_X

colors = ["r", "g" , "b", "c", "m", "y", "navy", "purple" ,"gold", "gray" , "orange", "lime", "pink","tomato" , "khaki", "azure" , "linen" , "rose", ]
plt.figure('附件四 Class', figsize = (5, 3), dpi = 300)
plt.xticks(range(0, 10000,1000), rotation = 45, fontsize = 5)
plt.yticks(fontsize = 5)
plt.xlabel('波段', fontsize = 5)
plt.ylabel('吸光度', fontsize = 5)
plt.grid(True)
for i in range(data_X.shape[0]):
    plt.plot(data_X.columns[2:], data_X.iloc[i,2:], color = colors[int(data0.iloc[i,1])], linewidth =0.5)
plt.show()

max_min = data_X.agg(lambda x: x.max() - x.min())
print(max_min.describe())
data_X1 = data_X.loc[:,max_min > max_min.mean()+0 * max_min.std()]#t data1 = data_X
print(data_X.shape, data_X1.shape)

data_X_know = data_X1.loc[data0['OP'] !=0]
data_X_unknow = data_X1.loc[data0['OP'] == 0]
print(data_X_know.shape, '\n', data_X_unknow.shape)
print(data_X_know.index, '\n', data_X_unknow.index)
data_y_know = data0.loc[data0['OP'] != 0, 'OP']
print(data_X_know.shape, data_y_know.shape, data_X_unknow.shape)
print(data_y_know.index)

from scipy.spatial import distance_matrix

dis = pd.DataFrame(distance_matrix(data_X_know.values, data_X_unknow.values), index =data_X_know.index, columns = data_X_unknow.index)
dis_group = pd.concat([data_y_know, dis], axis = 1)

print(dis_group.shape)
print(dis_group.columns)
print(dis_group)

min_index = dis_group.idxmin()
print(min_index)
res = [*zip(OP_null_index, dis_group.loc[min_index[1:], 'OP'])]
res

dict1= {i : []for i in range(17)}
for i in range(len(res)):
    dict1[res[i][1]].append(res[i][0])
for i in dict1.keys():
    print(i, dict1[i], len(dict1[i]))
