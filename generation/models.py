import torch.nn as nn


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_features, n_classes, kernel_size=1, stride=1, padding=0, bias=False, softmax=False,
                 cnn=False):
        super(AuxiliaryClassifier, self).__init__()
        if cnn:
            classes = [nn.Conv2d(in_features, n_classes, kernel_size, stride, padding, bias=bias)]
            validity = [nn.Conv2d(in_features, n_classes, kernel_size, stride, padding, bias=bias), nn.Sigmoid()]
        else:
            classes = [nn.Linear(in_features, n_classes)]
            validity = [nn.Linear(in_features, 1), nn.Sigmoid()]
        self.classes = nn.Sequential(*classes, nn.Softmax if softmax else Identity())
        self.validity = nn.Sequential(*validity)

    def forward(self, x):
        return self.classes(x), self.validity(x)


class Generator(nn.Module):
    def __init__(self, cnn=False):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        if not cnn:
            model = [nn.Linear(in_features=100, out_features=512), act, nn.Dropout(p=0.5)]
            model += [nn.Linear(in_features=512, out_features=256), act, nn.Dropout(p=0.5)]
            model += [nn.Linear(in_features=256, out_features=28 * 28), nn.Tanh()]
        else:
            norm = nn.BatchNorm2d
            model = [nn.Linear(100, 512 * 4 * 4), View([512, 4, 4]), norm(512), act]  # 4x4
            model += [nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2), norm(256), act]  # 7x7
            model += [nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1), norm(128), act]  # 14x14
            model += [nn.ConvTranspose2d(128, 1, 5, stride=2, padding=2,  output_padding=1), nn.Tanh()]  # 28x28
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Discriminator(nn.Module):
    def __init__(self, cnn=False):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        if not cnn:
            model = [nn.Linear(in_features=28 * 28, out_features=512), act]
            model += [nn.Linear(in_features=512, out_features=256), act]
            model += [AuxiliaryClassifier(256, 10)]
        else:
            norm = nn.BatchNorm2d
            model = [nn.Conv2d(1, 128, 5, stride=2, padding=2, bias=False), act]  # 14x14
            model += [nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False), norm(256), act]  # 7x7
            model += [nn.Conv2d(256, 512, 5, stride=2, padding=2, bias=False), norm(512), act]  # 4x4
            model += [AuxiliaryClassifier(512, 10, kernel_size=4, bias=False, cnn=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class View(nn.Module):
    def __init__(self, output_shape):
        super(View, self).__init__()
        self.output_shape = output_shape

    def forward(self, x):
        return x.view(x.shape[0], *self.output_shape)


def weights_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.detach().normal_(mean=0., std=0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1., 0.02)
        module.bias.detach().zero_()

    else:
        pass
