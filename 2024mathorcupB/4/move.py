# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:38:33 2024

@author: admin
"""

import os
import shutil

def move_files_to_single_folder(root_folder):
    # 获取root_folder下的所有文件夹
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    for subfolder in subfolders:
        # 获取二级文件夹下的所有文件
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        for file in files:
            # 移动文件到同一文件夹下
            shutil.move(file, root_folder)
            print(f'Moved {file} to {root_folder}')

# 根文件夹路径
root_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/4/4_Recognize/训练集'
move_files_to_single_folder(root_folder)
