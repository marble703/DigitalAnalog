# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:22:14 2024

@author: admin
"""

import os
import shutil
import random

def split_dataset(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    files = os.listdir(input_folder)
    random.shuffle(files)  # 打乱文件顺序，以确保随机性
    
    # 计算划分数量
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val
    
    # 划分文件
    train_files = files[:num_train]
    val_files = files[num_train:num_train + num_val]
    test_files = files[num_train + num_val:]
    
    # 将同名的txt文件和bmp文件分入同一个文件夹
    for file in train_files:
        if file.endswith('.txt'):
            if os.path.exists(os.path.join(input_folder, file.replace('.txt', '.bmp'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "train/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.txt', '.bmp')), os.path.join(output_folder, "train/", file.replace('.txt', '.bmp')))
        elif file.endswith('.bmp'):
            if os.path.exists(os.path.join(input_folder, file.replace('.bmp', '.txt'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "train/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.bmp', '.txt')), os.path.join(output_folder, "train/", file.replace('.bmp', '.txt')))
    for file in val_files:
        if file.endswith('.txt'):
            if os.path.exists(os.path.join(input_folder, file.replace('.txt', '.bmp'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "val/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.txt', '.bmp')), os.path.join(output_folder, "val/", file.replace('.txt', '.bmp')))
        elif file.endswith('.bmp'):
            if os.path.exists(os.path.join(input_folder, file.replace('.bmp', '.txt'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "val/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.bmp', '.txt')), os.path.join(output_folder, "val/", file.replace('.bmp', '.txt')))
    for file in test_files:
        if file.endswith('.txt'):
            if os.path.exists(os.path.join(input_folder, file.replace('.txt', '.bmp'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "test/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.txt', '.bmp')), os.path.join(output_folder, "test/", file.replace('.txt', '.bmp')))
        elif file.endswith('.bmp'):
            if os.path.exists(os.path.join(input_folder, file.replace('.bmp', '.txt'))):
                shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, "test/", file))
                shutil.copy(os.path.join(input_folder, file.replace('.bmp', '.txt')), os.path.join(output_folder, "test/", file.replace('.bmp', '.txt')))
    
    print("数据集划分完成：")
    print("训练集数量：", len(train_files))
    print("验证集数量：", len(val_files))
    print("测试集数量：", len(test_files))

# 调用函数划分数据集
input_folder = "D:/programme/2022mathorcupB/2024mathorcupB/4/4_Recognize/训练集"  # 输入文件夹路径
output_folder = "D:/programme/2022mathorcupB/2024mathorcupB/4/"  # 输出文件夹路径
split_dataset(input_folder, output_folder)

