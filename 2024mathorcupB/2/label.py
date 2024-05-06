import os
import json
import cv2

def convert_to_yolo_format(annotation, image_width, image_height):
    """
    将注释转换为YOLO接受的格式。
    """
    x_min, y_min, x_max, y_max, class_id = annotation
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

def validate_annotations(annotations, image_width, image_height):
    """
    验证注释是否在图像边界内。
    """
    for annotation in annotations:
        x_min, y_min, x_max, y_max, _ = annotation
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            return False
    return True

def create_annotation(json_folder, image_folder, output_folder):
    """
    根据文件夹中的每个JSON文件和相应的JPG文件创建同名的TXT标注文件。
    """
    os.makedirs(output_folder, exist_ok=True)
    for json_filename in os.listdir(json_folder):
        if json_filename.endswith('.json'):
            # 读取JSON文件
            json_filepath = os.path.join(json_folder, json_filename)
            with open(json_filepath, 'r') as json_file:
                data = json.load(json_file)
            img_name = data['img_name']
            annotations = data['ann']

            # 读取图像尺寸
            img_path = os.path.join(image_folder, img_name + '.jpg')
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read image '{img_name}'")
                continue
            image_height, image_width, _ = img.shape

            # 验证注释
            if not validate_annotations(annotations, image_width, image_height):
                print(f"Error: Annotations exceed image boundary in '{json_filename}'")
                continue

            # 创建TXT标注文件
            txt_filename = os.path.splitext(json_filename)[0] + '.txt'
            txt_filepath = os.path.join(output_folder, txt_filename)
            with open(txt_filepath, 'w') as txt_file:
                for annotation in annotations:
                    # 将注释转换为YOLO格式并写入TXT文件
                    yolo_annotation = convert_to_yolo_format(annotation, image_width, image_height)
                    txt_file.write(yolo_annotation + '\n')
            print(f"Created annotation file '{txt_filename}'")

# JSON文件夹和图像文件夹路径
json_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/train/'
image_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/train/'
# 输出标注文件夹路径
output_folder = 'D:/programme/2022mathorcupB/2024mathorcupB/2/train/'

# 创建标注文件
create_annotation(json_folder, image_folder, output_folder)
