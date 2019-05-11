import torch.nn as nn
import torch.nn.functional as F


class SEResidualNetwork(nn.Module):
    def __init__(self, n):
        super(SEResidualNetwork, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d

        network = []
        network += [nn.ZeroPad2d(1), nn.Conv2d(3, 16, 3, bias=False), norm(16), act]
        for _ in range(n):
            network += [SEResidualBlock(16, 16)]

        network += [SEResidualBlock(16, 32, first_conv_stride=2)]
        for _ in range(n - 1):
            network += [SEResidualBlock(32, 32)]

        network += [SEResidualBlock(32, 64, first_conv_stride=2)]
        for _ in range(n - 1):
            network += [SEResidualBlock(64, 64)]

        network += [nn.AdaptiveAvgPool2d(1), View(-1), nn.Linear(64, 10)]

        self.network = nn.Sequential(*network)

        self.apply(init_weight)

        print("Total parameters: {}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, x):
        return self.network(x)


class SEResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, first_conv_stride=1):
        super(SEResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        pad = nn.ZeroPad2d

        block = [pad(1), nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, bias=False), norm(output_ch), act]
        block += [pad(1), nn.Conv2d(output_ch, output_ch, 3, bias=False), norm(output_ch),
                  SqueezeExcitationBlock(output_ch, output_ch)]

        if input_ch != output_ch:
            self.varying_size = True
            """
            As far as I know, the original authors didn't mention about what down-sampling method they used in identity
            mapping. This can be max pooling or average pooling. Please give me an advice if anyone knows about this 
            pooling layer. For now, I'll use max pooling layer. Also, when I pad along the channel dimension, I add zero
            entries behind (not front) original data. This, best of my knowledge, is also not mentioned whether front or
            behind (or may be half and half) the original data across channel dimension. But I believe this is not a big
            issue.
            """
            side_block = [pad(1), nn.MaxPool2d(kernel_size=3, stride=2),
                          nn.ConstantPad3d((0, 0, 0, 0, 0, output_ch - input_ch), value=0.)]
            self.side_block = nn.Sequential(*side_block)

        else:
            self.varying_size = False

        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.varying_size:
            return F.relu(self.side_block(x) + self.block(x))

        else:
            return F.relu(x + self.block(x))


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, input_ch, output_ch, reduction_ratio=16, pooling='average'):
        super(SqueezeExcitationBlock, self).__init__()
        if pooling == 'average':
            pool = nn.AdaptiveAvgPool2d
        elif pooling == 'max':
            pool = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError(print("Invalid pooling {}. Please choose among ['average', 'max']."
                                            .format(pooling)))

        block = [pool(1), View(-1), nn.Linear(input_ch, input_ch // reduction_ratio), nn.ReLU(inplace=True),
                 nn.Linear(input_ch // reduction_ratio, output_ch), nn.Sigmoid(), View(output_ch, 1, 1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x * self.block(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


def init_weight(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity='relu')

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().fill_(1.)
        module.bias.detach().fill_(0.)
