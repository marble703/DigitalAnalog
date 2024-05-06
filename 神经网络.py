#利用Pytorch库构建神经网络模型，这里以训练拟合函数为例
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 准备训练数据
x = np.linspace(0, 2*np.pi, 1000)  # 生成一些x，这些x属于一个区间，并在这个区间内均匀分布
y = np.sin(x)  #我们希望拟合的函数

# 数据处理，将数据转换为 PyTorch 张量
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 神经网络模型，这里用3个隐层，每层神经元个数为6,7,6
class PolynomialModel(nn.Module):
    def __init__(self):
        super(PolynomialModel, self).__init__()
        self.fc1 = nn.Linear(1, 8)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(8, 40)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(40, 20)  # 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(20, 1)   # 第三个隐藏层到输出层

    def forward(self, x):
        x = torch.sigmoid(x)  # 使用 sigmoid 激活函数
        x = self.fc1(x)  # 第一个隐藏层的线性变换
        x = torch.sigmoid(x)  # 使用 sigmoid 激活函数
        x = self.fc2(x)  # 第二个隐藏层的线性变换
        x = torch.sigmoid(x)  # 使用 sigmoid 激活函数
        x = self.fc3(x)  # 第三个隐藏层的线性变换
        x = torch.sigmoid(x)  # 使用 sigmoid 激活函数
        x = self.fc4(x)  # 输出层的线性变换
        return x

# 实例化模型、损失函数和优化器
model = PolynomialModel()
criterion = nn.MSELoss()  # 损失函数，计算均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr即学习率，影响训练精度，每一次参数更新中沿梯度方向更新的距离

# 训练模型
N=int(input())  # 训练次数
epochs = N
for epoch in range(epochs):  # 迭代
    optimizer.zero_grad()  # 每次开始时先将梯度清零
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()  # 根据损失函数计算梯度
    optimizer.step()
    if ((epoch+1) % 100 == 0) and (epoch!=0):
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')  # 显示训练阶段和误差

# 绘制拟合曲线
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).detach().numpy()

plt.plot(x, y, label='True function')
plt.plot(x, predicted, label='Predicted function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting sin(x) in neural network model')
plt.legend()
plt.show()