import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 读取文件路径
file_path = 'map.csv'  # 请根据实际路径调整
file_path2 = 'agv.csv'

# 创建无向图
G = nx.Graph()

# 用于存储节点类型的字典
node_types = {}

# 读取文件并解析节点信息
with open(file_path, 'r') as file:
    next(file)  # 跳过标题行
    for line in file:
        parts = line.strip().split(',')
        node_type = parts[0]  # 节点类型
        node_id = parts[1] + ',' + parts[2]  # 使用X,Y坐标作为唯一标识符
        x, y = float(parts[1]), float(parts[2])  # 节点坐标
        neighbors = parts[3]  # 邻居节点

        # 添加节点，并记录节点类型
        G.add_node(node_id, pos=(x, y), type=node_type)
        node_types[node_id] = node_type

        # 添加边
        if neighbors:
            for neighbor in neighbors.split(';'):
                if neighbor:  # 确保邻居信息不为空
                    neighbor_x, neighbor_y = neighbor.split(':')
                    neighbor_id = neighbor_x + ',' + neighbor_y
                    G.add_edge(node_id, neighbor_id)

#添加agv的坐标
with open(file_path2, 'r') as file:
    next(file)  # 跳过标题行
    for line in file:
        parts = line.strip().split(',')
        node_type = "8"  # 节点类型
        node_id = parts[1] + ',' + parts[2]  # 使用X,Y坐标作为唯一标识符
        x, y = float(parts[1]), float(parts[2])  # 节点坐标
        # 添加节点，并记录节点类型
        G.add_node(node_id, pos=(x, y), type=node_type)
        node_types[node_id] = node_type

# 为不同的节点类型分配颜色
type_colors = {"*": "white", "1": "gray", "2": "green", "3": "yellow", "4": "black","5":"blue","6":"pink","7":"red"}
default_color = "orange"  # 未预定义类型的默认颜色

# 生成节点颜色列表，使用节点类型对应的颜色或默认颜色
node_colors = [type_colors.get(node_types[node], default_color) for node in G]

# 使用节点位置和类型颜色绘制图
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, edge_color='gray')

plt.show()  # 显示图
