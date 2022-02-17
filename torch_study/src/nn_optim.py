import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 数据集
dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 加载数据集
dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()

# 设置优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data
        output = tudui(imgs)
        result_loss = loss(output, target)
        # print(result_loss)
        # 梯度清零
        optim.zero_grad()
        # 反向传播
        result_loss.backward()
        # 优化梯度
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
