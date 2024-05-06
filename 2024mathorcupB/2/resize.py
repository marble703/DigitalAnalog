# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:57:48 2024

@author: admin
"""

import os
import cv2

def resize_images(input_folder, output_folder, target_size):
    """
    将输入文件夹中的图像调整为目标大小，并保存到输出文件夹。
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image '{filename}'")
                continue
            # 调整大小
            resized_image = cv2.resize(image, target_size)
            # 保存调整大小后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)
            print(f"Resized '{filename}' to {target_size}")

# 输入文件夹和输出文件夹路径
input_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/val/'
output_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/val/'
# 目标大小
target_size = (416, 416)  # 你可以根据需要修改目标大小

# 调整图像大小
resize_images(input_folder, output_folder, target_size)
