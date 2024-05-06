import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']= ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data0 = pd.read_csv(r'附件1.csv', index_col = 0)
data0.shape

#data0.head()

#data0.describe()

#data0.info()

#data0.isnull().any().any()

def my_plot(x):
    plt.plot(x.index, x.values, linewidth = 0.5)

def PlotSpectrum(data, strO):
    fontsize = 5
    plt.figure(strO, figsize = (5, 3), dpi= 300)
    plt.xticks(range(0, 4001, 500), rotation = 45, fontsize = 5)
    plt.yticks(fontsize = fontsize)
    plt.xlabel('波段', fontsize = 6)
    plt.ylabel('吸光度', fontsize = 6)
    plt.grid(True)
    data.agg(lambda x: my_plot(x), axis = 1)
    plt.show()

#PlotSpectrum(data0,'附件一光谱数据曲线图')

def box(x):
    small=x.mean()-3*x.std()
    large=x.mean()+3*x.std()
    return (x<small)|(x>large)

yczhi = data0.agg(lambda x: box(x))
yczhi_index = data0[(yczhi.sum(axis = 1) > 100)].index
print(yczhi_index)

data0.drop(yczhi_index, axis = 0, inplace = True)
data0.shape

PlotSpectrum(data0, '附件一光谱数据曲线图')

data_corr = data0.corr()
data_corr

data_std = data0.std()
print(data_std.min(), data_std.max())
(data_std<0.05).sum()

max_min = data0.agg(lambda x: x.max() - x.min())
print(max_min.describe())
data1 = data0.loc[:,max_min > max_min.mean()+0 * max_min.std()]
data1.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data1)
data2 = scaler.transform(data1)
data2 = pd.DataFrame(data1, index = data1.index)

PlotSpectrum(data1, '附件一光谱标准化处理后数据曲线图')

from sklearn.manifold import Isomap
isomap = Isomap(n_components = 3).fit(data1)
data2 = isomap.transform(data1)
print(data2.shape)
plt.figure()
plt.scatter(data2[:,0], data2[:,1])
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scores = []

for n_clusters in range(2, 8):
    cluster = KMeans(n_clusters = n_clusters, random_state = 0).fit(data1)
    score = silhouette_score(data1, cluster.labels_)
    scores.append(score)

print(scores)
plt.figure('聚类数量的轮廓系数')
plt.plot(range(2, 8), scores, '-o')
plt.show()

cluster = KMeans(n_clusters = 3).fit(data1)
pd.Series(cluster.labels_).value_counts().sort_index()

plt.figure()
plt.scatter(data2[:,0], data2[:,1], c = cluster.labels_)
plt.show()

colors = {'A': 'r', 'B': 'g', 'C': 'b', 'D': 'y'}
colors = ['r', 'g', 'b']
plt.figure('附件一分类图', figsize = (5, 3), dpi = 300)
plt.xticks(range(0,10000,1000), rotation = 45, fontsize = 5)
plt.yticks(fontsize = 5)
plt.xlabel('波段', fontsize = 5)
plt.ylabel('吸光度', fontsize = 5)
for i in range(data0.shape[0]):
    plt.plot(data0.columns, data0.iloc[i,:], c = colors[cluster.labels_[i]], linewidth = 0.5)
plt.grid(True)
plt.show()

