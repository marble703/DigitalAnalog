# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:37:29 2024

@author: admin
"""

import os

# 示例用法
dataset_folder = "D:/programme/2022mathorcupB/2024mathorcupB/4/4_Recognize/训练集"
output_file = "class_names.txt"

# 获取数据集文件夹下所有文件夹（类别）
categories = sorted(os.listdir(dataset_folder))

# 将类别写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, category in enumerate(categories):
        f.write(f"{idx}: {category}\n")

        # 遍历当前类别文件夹下的 bmp 图片文件
        category_folder = os.path.join(dataset_folder, category)
        for filename in sorted(os.listdir(category_folder)):
            if filename.endswith(".bmp"):
                # 创建同名的 TXT 文件，并写入指定内容
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(category_folder, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(str(idx) + " 0.5 0.5 1.0 1.0\n")