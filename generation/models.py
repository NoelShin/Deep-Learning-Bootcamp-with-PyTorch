import torch.nn as nn


def weights_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.detach().normal_(mean=0., std=0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1., 0.02)
        module.bias.detach().zero_()

    else:
        pass


class View(nn.Module):
    def __init__(self, output_shape):
        super(View, self).__init__()
        self.output_shape = output_shape

    def forward(self, x):
        return x.view(x.shape[0], *self.output_shape)


class Generator(nn.Module):
    def __init__(self, dataset_name):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d

        if dataset_name == 'CIFAR10':  # Input shape 3x32x32
            model = [nn.Linear(100, 512 * 4 * 4), View([512, 4, 4]), norm(512), act]  # 4x4
            model += [nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1), norm(256), act]  # 8x8
            model += [nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1), norm(128), act]  # 16x16
            model += [nn.ConvTranspose2d(128, 3, 5, stride=2, padding=2, output_padding=1), nn.Tanh()]  # 32x32

        elif dataset_name == 'LSUN':  # Input shape 3x64x64
            model = [nn.Linear(100, 1024 * 4 * 4), View([1024, 4, 4]), norm(1024), act]  # 4x4
            model += [nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), norm(512), act]  # 8x8
            model += [nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1), norm(256), act]  # 16x16
            model += [nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1), norm(128), act]  # 32x32
            model += [nn.ConvTranspose2d(128, 3, 5, stride=2, padding=2, output_padding=1), nn.Tanh()]  # 64x64

        elif dataset_name == 'MNIST':  # Input shape 1x28x28
            model = [nn.Linear(100, 256 * 4 * 4), View([256, 4, 4]), norm(256), act]  # 4x4
            model += [nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2), norm(128), act]  # 7x7
            model += [nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1), norm(64), act]  # 14x14
            model += [nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh()]  # 28x28

        else:
            raise NotImplementedError

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, dataset_name):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        norm = nn.BatchNorm2d

        if dataset_name == 'CIFAR10':  # Input shape 3x32x32
            model = [nn.Conv2d(3, 128, 5, stride=2, padding=2, bias=False), act]  # 16x16
            model += [nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False), norm(128), act]  # 8x8
            model += [nn.Conv2d(256, 512, 5, stride=2, padding=2, bias=False), norm(256), act]  # 4x4
            model += [nn.Conv2d(512, 1, 4, stride=2, padding=2, bias=False), nn.Sigmoid()]  # 1x1

        elif dataset_name == 'LSUN':  # Input shape 3x64x64
            model = [nn.Conv2d(3, 128, 5, stride=2, padding=2, bias=False), act]  # 128x32x32
            model += [nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False), norm(128), act]  # 256x16x16
            model += [nn.Conv2d(256, 512, 5, stride=2, padding=2, bias=False), norm(256), act]  # 512x8x8
            model += [nn.Conv2d(512, 1024, 5, stride=2, padding=2, bias=False), norm(512), act]  # 1024x4x4
            model += [nn.Conv2d(1024, 1, 4), nn.Sigmoid()]  # 1x1x1

        elif dataset_name == 'MNIST':  # Input shape 1x28x28
            model = [nn.Conv2d(1, 64, 5, stride=2, padding=2, bias=False), act]  # 14x14
            model += [nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False), norm(128), act]  # 7x7
            model += [nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False), norm(256), act]  # 4x4
            model += [nn.Conv2d(256, 1, 4, bias=False), nn.Sigmoid()]  # 1x1

        else:
            raise NotImplementedError

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
