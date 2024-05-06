import os
import cv2

def resize_images_with_padding(input_folder, output_folder, target_size):
    """
    将输入文件夹中的图像调整为目标大小，并在调整过程中保持原始图像的比例，填充黑色背景。
    同时生成同名的YOLO格式标注文件。
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.bmp')):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image '{filename}'")
                continue
            
            # 计算调整大小后的图像
            h, w, _ = image.shape
            max_dim = max(h, w)
            scale = max_dim / max(target_size)
            new_h = int(h / scale)
            new_w = int(w / scale)
            pad_h = (target_size[1] - new_h) // 2
            pad_w = (target_size[0] - new_w) // 2
            
            # 调整大小并填充黑色背景
            resized_image = cv2.resize(image, (new_w, new_h))
            padded_image = cv2.copyMakeBorder(resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # 保存调整大小后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_image)
            print(f"Resized '{filename}' with padding to {target_size}")
            
            # 生成YOLO格式的标注文件
            with open(os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt'), 'w') as f:
                f.write(f"0 {(pad_w + new_w / 2) / target_size[0]} {(pad_h + new_h / 2) / target_size[1]} {new_w / target_size[0]} {new_h / target_size[1]}")

# 输入文件夹和输出文件夹路径
input_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/4/train/'
output_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/4/train_padded/'
# 目标大小
target_size = (416, 416)  # 你可以根据需要修改目标大小

# 调整图像大小并填充黑色背景
resize_images_with_padding(input_folder, output_folder, target_size)
