## 数据处理流程
（我不知道怎么把链接显示关掉）
#### 1.运行 divide_file.py 
将数据集划分为训练集，验证集，测试集

#### 2.运行 label.py和change_label.py
创建yolo格式的标注文件（需要对每个数据集分别运行一遍）

#### 3.运行 resize.py（可选）
修改所有图像大小（需要对每个数据集分别运行一遍） 

## 训练流程

#### 1.下载yolov5，安装requirement.txt需要的包
不会的话网上有教程

#### 2.将mcb.yaml复制到yolov5的data文件夹下，yolov5s.yaml复制到model下。
记得修改里面的路径
#### 3.运行命令
python train.py --img 416 --batch 16 --epochs 50 --data data/mydata/mcb.yaml --cfg models/yolov5s.yaml