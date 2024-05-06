# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:55:28 2024

@author: admin
"""

import os

def modify_annotation_files(annotation_folder):
    """
    将标注文件中的所有标签设置为0。
    """
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(annotation_folder, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w') as file:
                for line in lines:
                    # 将标签设置为0并写回文件
                    file.write('0' + line[3:])

# 标注文件夹路径
annotation_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/val/'

# 修改标注文件
modify_annotation_files(annotation_folder)
