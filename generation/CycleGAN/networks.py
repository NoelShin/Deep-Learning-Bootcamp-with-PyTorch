import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        """
        1. The authors used 70x70 (receptive field size) patch GAN. The structures can be varied with your own purpose.
        2. Discriminator does NOT take a condition image which is fed to the generator.
        3. No normalization layer is applied after the first convolution layer.
        4. As the authors used LSGAN for stable training, no sigmoid activation is attached at the last layer.
        """
        act = nn.LeakyReLU(0.2, inplace=True)
        in_channels = opt.out_channels
        n_df = opt.n_df
        norm = nn.InstanceNorm2d

        network = [nn.Conv2d(in_channels, n_df, 4, stride=2, padding=1), act]
        network += [nn.Conv2d(n_df, 2 * n_df, 4, stride=2, padding=1), norm(2 * n_df), act]
        network += [nn.Conv2d(2 * n_df, 4 * n_df, 4, stride=2, padding=1), norm(4 * n_df), act]
        network += [nn.Conv2d(4 * n_df, 8 * n_df, 4, padding=1), norm(8 * n_df), act]
        network += [nn.Conv2d(8 * n_df, 1, 4, padding=1)]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        """
        1. The authors used n_RB = 6 for 128x128 and 9 for 256x256 or higher image size. You can change n_RB in the
           options.py.
        2. No normalization layer is applied after the last convolution layer.
        """
        act = nn.ReLU(inplace=True)
        in_channels = opt.in_channels
        out_channels = opt.out_channels
        n_gf = opt.n_gf
        n_RB = opt.n_RB
        norm = nn.InstanceNorm2d

        network = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, n_gf, 7), norm(n_gf), act]
        network += [nn.Conv2d(n_gf, 2 * n_gf, 3, stride=2, padding=1), norm(2 * n_gf), act]
        network += [nn.Conv2d(2 * n_gf, 4 * n_gf, 3, stride=2, padding=1), norm(4 * n_gf), act]

        for block in range(n_RB):
            network += [ResidualBlock(4 * n_gf)]

        network += [nn.ConvTranspose2d(4 * n_gf, 2 * n_gf, 3, stride=2, padding=1, output_padding=1), norm(2 * n_gf),
                    act]
        network += [nn.ConvTranspose2d(2 * n_gf, n_gf, 3, stride=2, padding=1, output_padding=1), norm(n_gf), act]
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


def update_lr(init_lr, old_lr, n_epoch_decay, *optims):
    delta_lr = init_lr / n_epoch_decay
    new_lr = old_lr - delta_lr

    for optim in optims:
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr


def weights_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.detach().normal_(mean=0., std=0.02)
