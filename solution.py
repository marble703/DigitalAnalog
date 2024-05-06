import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix

import sys

# 读取地图数据
def read_map_data(map_file):
    """
    读取地图数据并返回地图信息。

    Args:
        map_file (str): 地图数据文件路径。

    Returns:
        dict: 包含地图信息的字典，键为坐标元组 (x, y)，值为与该坐标相邻的节点及对应的距离的字典。
    """
    map_data = {}
    # 读取地图数据文件为 DataFrame
    df = pd.read_csv("map.csv")
    # 跳过标题行
    df = df[1:]
    for i in range(1,1 + df.index.size):
        x, y = df[' X'][i], df[' Y'][i]
        neighbors = {}
        # 处理邻居节点数据
        pre_data = df[df.columns[3]][i]
        if(str(pre_data) == "nan"):
            neighbor_data = ['99:99']
        else:
            neighbor_data = df[df.columns[3]][i].split(';')
        for i in neighbor_data:
            parts = i.split(':')
            x = int(parts[0])
            y = int(parts[1])
            data = (x, y)
            neighbors[data] = 1
    
        map_data[(x, y)] = neighbors
        
    return map_data


def read_agv_data(agv_file):
    """
    读取AGV初始位置数据并返回AGV位置信息。

    Args:
        agv_file (str): AGV初始位置数据文件路径。

    Returns:
        dict: 包含AGV位置信息的字典，键为AGV编号，值为对应的初始坐标 (x, y)。
    """
    agv_positions = {}
    # 读取AGV数据文件为 DataFrame
    df = pd.read_csv(agv_file)
    # 跳过标题行
    df = df.iloc[1:]
    for index, row in df.iterrows():
        agv_id = int(row['#AGV_ID'])
        x = int(row['X'])
        y = int(row['Y'])
        agv_positions[agv_id] = (x, y)
    return agv_positions


# 读取最短路径矩阵
def read_distance_matrix(dis_file):
    """
    读取最短路径矩阵并返回矩阵数据。

    Args:
        dis_file (str): 最短路径矩阵数据文件路径。

    Returns:
        pandas.DataFrame: 包含最短路径矩阵数据的 DataFrame。
    """
    return pd.read_csv(dis_file)

# 初始化种群
def initialize_population(agv_count, map_data):
    """
    初始化种群，每个个体为一个AGV的路径，起始位置为随机选择的地图节点。

    Args:
        agv_count (int): AGV数量。
        map_data (dict): 地图数据。

    Returns:
        list: 包含多个个体（路径）的种群。
    """
    population = []
    for _ in range(agv_count):
        agv_path = [random.choice(list(map_data.keys()))]  # AGV初始位置
        population.append(agv_path)
    return population

# 计算适应度
def fitness_function(path, dis_matrix):
    """
    计算路径的适应度，即路径的总距离。

    Args:
        path (list): 路径，由地图节点坐标组成的列表。
        dis_matrix (pandas.DataFrame): 最短路径矩阵。

    Returns:
        int: 路径的总距离（适应度）。
    """
    total_distance = 0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        total_distance += dis_matrix.loc[current_node, next_node]
    return total_distance

# 交叉操作
def crossover(parent1, parent2):
    """
    交叉操作，生成两个子代。

    Args:
        parent1 (list): 第一个父代（路径）。
        parent2 (list): 第二个父代（路径）。

    Returns:
        tuple: 包含两个子代（路径）的元组。
    """
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1, parent2

    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# 变异操作
def mutate(path, map_data):
    """
    变异操作，对路径进行变异。

    Args:
        path (list): 路径。
        map_data (dict): 地图数据。

    Returns:
        list: 变异后的路径。
    """
    if len(path) <= 2:
        return path

    mutated_path = path[:]
    mutation_point = random.randint(1, len(path) - 2)
    new_node = random.choice(list(map_data[path[mutation_point]].keys()))
    mutated_path[mutation_point] = new_node
    return mutated_path


# 遗传算法主函数
def genetic_algorithm(population, dis_matrix, map_data, max_generations=1000):
    """
    遗传算法主函数。

    Args:
        population (list): 初始种群。
        dis_matrix (pandas.DataFrame): 最短路径矩阵。
        map_data (dict): 地图数据。
        max_generations (int): 最大迭代次数，默认为1000。

    Returns:
        int: 最优解（最短路径）的总距离，如果种群为空则返回 None。
    """
    for generation in range(max_generations):
        # 选择
        sorted_population = sorted(population, key=lambda x: fitness_function(x, dis_matrix))
        selected_parents = sorted_population[:len(population) // 2]

        # 生成子代
        offspring = []
        for i in range(0, len(selected_parents)-1, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # 变异
        for i in range(len(offspring)):
            if random.random() < 0.1:  # 变异率
                offspring[i] = mutate(offspring[i], map_data)

        population = offspring

    # 检查种群是否为空
    if not population:
        return None

    # 计算最优解的总距离
    best_path = sorted(population, key=lambda x: fitness_function(x, dis_matrix))[0]
    best_distance = fitness_function(best_path, dis_matrix)
    return best_distance


# 主函数
if __name__ == "__main__":
    # 文件路径
    map_file = "map.csv"
    agv_file = "agv.csv"
    dis_file = "dis.csv"

    # 读取数据
    map_data = read_map_data(map_file)
    agv_positions = read_agv_data(agv_file)
    dis_matrix = read_distance_matrix(dis_file)

    # 初始化种群
    population = initialize_population(len(agv_positions), map_data)

    # 遗传算法求解
    best_distance = genetic_algorithm(population, dis_matrix, map_data)

    # 输出最优解的距离
    print(best_distance)

