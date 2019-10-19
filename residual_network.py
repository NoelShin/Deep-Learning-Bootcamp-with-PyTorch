import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        return self.act(x + y)


class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()
        network = []
        network += [nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    ResidualBlock(16),
                    ResidualBlock(16),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    ResidualBlock(32),
                    ResidualBlock(32),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    ResidualBlock(64),
                    ResidualBlock(64),
                    nn.AdaptiveAvgPool2d((1, 1)),  # 64x64x1x1
                    View(64),  # 64x64
                    nn.Linear(64, 10)]  # 64x10
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # 64x64x1x1 >> 64x64