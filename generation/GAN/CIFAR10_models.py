import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [nn.Linear(in_features=100, out_features=512), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        model += [nn.Linear(in_features=512, out_features=256), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        model += [nn.Linear(in_features=256, out_features=3 * 32 * 32), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [nn.Linear(in_features=3 * 32 * 32, out_features=512), nn.LeakyReLU(inplace=True, negative_slope=0.2)]
        model += [nn.Linear(in_features=512, out_features=256), nn.LeakyReLU(inplace=True, negative_slope=0.2)]
        model += [nn.Linear(in_features=256, out_features=3), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
