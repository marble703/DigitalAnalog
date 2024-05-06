# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:00:06 2024

@author: admin
"""
import os
import re
import random
import string

def replace_non_english_characters(folder_path):
    try:
        # 遍历文件夹下的所有文件并按照文件名排序
        files = sorted(os.listdir(folder_path))
        processed_files = set()  # 用集合来存储已处理过的文件名，避免多次处理同一个文件名
        for filename in files:
            if filename in processed_files:
                continue
            basename, extension = os.path.splitext(filename)
            # 检查文件名是否包含非英文或数字的字符
            if re.search(r'[^a-zA-Z0-9.]', basename):
                # 获取同名但后缀不同的文件名
                same_name_files = [f for f in files if f.startswith(basename) and os.path.splitext(f)[1] != extension]
                # 生成随机的英文字符替换非英文或数字的字符
                new_basename = ''.join(random.choices(string.ascii_letters + string.digits, k=len(basename)))
                new_filename = new_basename + extension
                # 重命名同名但后缀名不同的文件
                for file in same_name_files:
                    try:
                        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_basename + os.path.splitext(file)[1]))
                        print(f'Renamed {file} to {new_basename + os.path.splitext(file)[1]}')
                    except FileNotFoundError:
                        print(f'File {file} not found')
                # 重命名当前文件
                try:
                    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
                    print(f'Renamed {filename} to {new_filename}')
                except FileNotFoundError:
                    print(f'File {filename} not found')
            processed_files.add(filename)
    except FileNotFoundError:
        print(f'Folder {folder_path} not found')

# 文件夹路径
folder_path = 'D:/programme/2022mathorcupB/2024mathorcupB/4/4_Recognize/训练集/'
replace_non_english_characters(folder_path)
