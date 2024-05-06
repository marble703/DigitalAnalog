import cv2
import numpy as np

def preprocess_image(image):
    # 二值化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    # 手动设置二值化阈值
    threshold_value = 150  # 调整阈值的值
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 伽马矫正
    gamma = 1.5
    binary = np.uint8(cv2.pow(binary / 255.0, gamma) * 255.0)
    
    # 连通组件标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    kernel = np.ones((3, 3), np.uint8)
    # 腐蚀 
    binary = cv2.erode(binary, kernel, iterations=1) 
    
    kernel = np.ones((2, 2), np.uint8)
    # 膨胀
    binary = cv2.dilate(binary, kernel, iterations=1)
    # 获取连通域的面积
    areas = stats[:, cv2.CC_STAT_AREA]
    
    # 定义一个阈值，过滤掉较小的连通域
    min_area_threshold = 100  # 根据需要调整阈值大小
    
    # 遍历所有连通域，将面积小于阈值的连通域填充为白色
    for i in range(1, num_labels):
        if areas[i] < min_area_threshold:
            binary[labels == i] = 255
        
    
    # 使用中值滤波器降噪
    binary = cv2.medianBlur(binary, 3)  # 这里的参数5表示滤波器的尺寸，可以根据需要调整
    
    # 使用Canny边缘检测算法提取边缘
    binary2 = cv2.Canny(gray, 50, 150, apertureSize=3)

    return binary

# 读取图片
image_path1 = "h02060.jpg"  # 替换为你的图片路径
image_path2 = "w01637.jpg"  # 替换为你的图片路径
image_path3 = "w01870.jpg"  # 替换为你的图片路径

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)
image3 = cv2.imread(image_path3)

# 预处理图片
processed_image1 = preprocess_image(image1)
processed_image2 = preprocess_image(image2)
processed_image3 = preprocess_image(image3)

# 显示结果
cv2.imshow("Original Image", image1)
cv2.imshow("Processed Image", processed_image1)

output_path1 = "h02060_P.jpg"  # 替换为你想要保存的路径及文件名
output_path2 = "w01637_P.jpg"  # 替换为你想要保存的路径及文件名
output_path3 = "w01870_P.jpg"  # 替换为你想要保存的路径及文件名

cv2.imwrite(output_path1, processed_image1)
cv2.imwrite(output_path2, processed_image2)
cv2.imwrite(output_path3, processed_image3)

cv2.waitKey(0)
cv2.destroyAllWindows()
