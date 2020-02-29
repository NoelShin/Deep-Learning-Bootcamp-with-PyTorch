import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        in_channels = 3
        n_df = 64
        norm = nn.InstanceNorm2d

        network = [nn.Conv2d(in_channels, n_df, kernel_size=4, stride=2, padding=1), act]
        network += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, stride=2, padding=1), norm(2 * n_df), act]
        network += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, stride=2, padding=1), norm(4 * n_df), act]
        network += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1), norm(8 * n_df), act]
        network += [nn.Conv2d(8 * n_df, 1, 4, padding=1)]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        in_channels = 3
        out_channels = 3

        n_gf = 64
        n_RB = 6
        norm = nn.InstanceNorm2d
        # 128 x 128
        network = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, n_gf, kernel_size=7), norm(n_gf), act]
        network += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, stride=2, padding=1), norm(2 * n_gf), act]

        # 64 x 64
        # [i - k + 2 * p / s] + 1
        # [64 - 3 + 2 * 1 / s ] + 1 >>  [63 / 2] + 1  >> 32
        network += [nn.Conv2d(2 * n_gf, 4 * n_gf, kernel_size=3, stride=2, padding=1), norm(4 * n_gf), act]

        for i in range(n_RB):
            network += [ResidualBlock(4 * n_gf)]

        network += [nn.ConvTranspose2d(4 * n_gf, 2 * n_gf, 3, stride=2, padding=1, output_padding=1), norm(2 * n_gf),
                    act]
        network += [nn.ConvTranspose2d(4 * n_gf, 2 * n_gf, 3, stride=2, padding=1, output_padding=1), norm(2 * n_gf),
                    act]
        network += [nn.ReflectionPad2d(3), nn.Conv2d(n_gf, out_channels, 7), nn.Tanh()]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.InstanceNorm2d

        block = [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, n_ch, 3), norm(n_ch), act]
        block += [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, n_ch, 3), norm(n_ch)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


def weights_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.detach().normal_(mean=0., std=0.02)