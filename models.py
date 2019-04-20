import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        norm = nn.BatchNorm2d
        n_df = 64

        model = [nn.Conv2d(3, n_df, kernel_size=4, stride=2, padding=1, bias=False), act]
        model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, stride=2, padding=1, bias=False), norm(2 * n_df), act]
        model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, stride=2, padding=1, bias=False), norm(4 * n_df), act]
        model += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, stride=2, padding=1, bias=False), norm(8 * n_df), act]
        model += [nn.Conv2d(8 * n_df, 1, kernel_size=4, bias=False), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        act_down = nn.LeakyReLU(0.2, inplace=True)
        act_up = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        n_gf = 64

        # Input shape 1x128x128
        # Output shape 3x128x128
        self.down_1 = nn.Sequential(nn.Conv2d(1, n_gf, kernel_size=4, stride=2, padding=1, bias=False))   # 64x64x64
        self.down_2 = nn.Sequential(act_down, nn.Conv2d(n_gf, 2 * n_gf, kernel_size=4, stride=2, padding=1, bias=False),
                                    norm(2 * n_gf))  # 128x32x32
        self.down_3 = nn.Sequential(act_down, nn.Conv2d(2 * n_gf, 4 * n_gf, kernel_size=4, stride=2, padding=1,
                                                        bias=False), norm(4 * n_gf))  # 256x16x16
        self.down_4 = nn.Sequential(act_down, nn.Conv2d(4 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                        bias=False), norm(8 * n_gf))  # 512x8x8
        self.down_5 = nn.Sequential(act_down, nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                        bias=False), norm(8 * n_gf))
        self.down_6 = nn.Sequential(act_down, nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                        bias=False), norm(8 * n_gf))
        self.down_up = nn.Sequential(act_down,
                                     nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1, bias=False),
                                     act_up,
                                     nn.ConvTranspose2d(8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                        bias=False), norm(8 * n_gf), nn.Dropout2d(0.5))  # 2x2
        self.up_6 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * 8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                     bias=False),
                                  norm(8 * n_gf), nn.Dropout(0.5))  # 4x4 2 ** (8 - 6)
        self.up_5 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * 8 * n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1,
                                                     bias=False),
                                  norm(8 * n_gf), nn.Dropout(0.5))  # 8x8
        self.up_4 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * 8 * n_gf, 4 * n_gf, kernel_size=4, stride=2, padding=1,
                                                     bias=False),
                                  norm(4 * n_gf))  # 16x16
        self.up_3 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * 4 * n_gf, 2 * n_gf, kernel_size=4, stride=2, padding=1,
                                                     bias=False),
                                  norm(2 * n_gf))  # 32x32
        self.up_2 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * 2 * n_gf, n_gf, kernel_size=4, stride=2, padding=1,
                                                     bias=False),
                                  norm(n_gf))  # 64x64
        self.up_1 = nn.Sequential(act_up,
                                  nn.ConvTranspose2d(2 * n_gf, 3, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.Tanh())

    def forward(self, x):
        intermediate_layers = [x]
        intermediate_layers += [self.down_1(intermediate_layers[-1])]
        intermediate_layers += [self.down_2(intermediate_layers[-1])]
        intermediate_layers += [self.down_3(intermediate_layers[-1])]
        intermediate_layers += [self.down_4(intermediate_layers[-1])]
        intermediate_layers += [self.down_5(intermediate_layers[-1])]
        intermediate_layers += [self.down_6(intermediate_layers[-1])]
        x = self.down_up(intermediate_layers[-1])
        x = self.up_6(torch.cat((intermediate_layers[-1], x), dim=1))
        x = self.up_5(torch.cat((intermediate_layers[-2], x), dim=1))
        x = self.up_4(torch.cat((intermediate_layers[-3], x), dim=1))
        x = self.up_3(torch.cat((intermediate_layers[-4], x), dim=1))
        x = self.up_2(torch.cat((intermediate_layers[-5], x), dim=1))
        x = self.up_1(torch.cat((intermediate_layers[-6], x), dim=1))
        return x
