# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:44:23 2024

@author: admin
"""
import os
import pandas as pd

output_excel_file = "D:/programme/2022mathorcupB/2024mathorcupB/3/Test_results.xlsx"

# 文件夹路径
folder_path = 'D:/programme/2022mathorcupB/2024mathorcupB/3/exp9/labels/'

# 初始化空列表，用于存储所有文件的内容
data = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否是txt文件
    if file_name.endswith('.txt'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 获取对应的 jpg 文件名
        jpg_file_name = file_name.replace('.txt', '.jpg')
        jpg_file_path = os.path.join(folder_path, jpg_file_name)

        # 打开文件并逐行读取内容
        with open(file_path, 'r') as file:
            file_data = [line.strip().split() for line in file.readlines()]  # 去除每行末尾的换行符，并将字符串拆分为列表
            # 将每行第一个数据项修改为1.0
            formatted_data = [[float(item) if idx != 0 else 1.0 for idx, item in enumerate(line)] for line in file_data]
        # 将文件名及其格式化后的内容添加到数据列表中
        formatted_data = str(formatted_data)[1:-1]
        data.append({'file': file_name, 'context': formatted_data})

# 使用 pandas 创建 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 写入 Excel 文件
df.to_excel(output_excel_file, index=False)