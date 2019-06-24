import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DenseBlock(nn.Module):
    def __init__(self, n_layers, n_ch, growth_rate, bottleneck=True, efficient=True):
        super(DenseBlock, self).__init__()
        for i in range(n_layers):
            self.add_module('Dense_layer_{:d}'.format(i),
                            DenseLayer(n_ch + i * growth_rate, growth_rate, bottleneck, efficient))

        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers):
            x = getattr(self, 'Dense_layer_{:d}'.format(i))(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, n_ch, growth_rate, bottleneck=True, efficient=True):
        super(DenseLayer, self).__init__()
        layer = []
        if bottleneck:
            layer += [nn.BatchNorm2d(n_ch),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(n_ch, 4 * growth_rate, 1, bias=False)]
            layer += [nn.BatchNorm2d(4 * growth_rate),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)]
        else:
            layer += [nn.BatchNorm2d(n_ch),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(n_ch, growth_rate, 3, padding=1, bias=False)]

        self.layer = nn.Sequential(*layer)

        self.efficient = efficient

    def function(self, *inputs):
        return self.layer(torch.cat(inputs, dim=1))

    def forward(self, *inputs):
        if self.efficient and any(input.requires_grad for input in inputs):
            x = checkpoint(self.function, *inputs)
        else:
            x = self.layer(torch.cat(inputs, dim=1))
        return torch.cat((*inputs, x), dim=1)


class DenseNet(nn.Module):
    def __init__(self, depth, growth_rate, efficient=True, input_ch=3, n_classes=100):
        super(DenseNet, self).__init__()

        """
        Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output
        channel is performed on the input images.
        """
        assert depth in [40, 100]
        n_layers = 6 if depth == 40 else 16
        init_ch = 16
        network = [nn.Conv2d(input_ch, init_ch, 3, padding=1, bias=False)]

        network += [DenseBlock(n_layers=n_layers, n_ch=init_ch, growth_rate=growth_rate, bottleneck=False,
                               efficient=efficient)]
        n_ch = init_ch + growth_rate * n_layers

        network += [TransitionLayer(n_ch, compress_factor=1)]
        network += [DenseBlock(n_layers=n_layers, n_ch=n_ch, growth_rate=growth_rate, bottleneck=False,
                               efficient=efficient)]
        n_ch = n_ch + growth_rate * n_layers

        network += [TransitionLayer(n_ch, compress_factor=1)]
        network += [DenseBlock(n_layers=n_layers, n_ch=growth_rate, growth_rate=growth_rate, bottleneck=False,
                               efficient=efficient)]
        n_ch = n_ch + growth_rate * n_layers

        network += [nn.BatchNorm2d(n_ch),
                    nn.ReLU(True),
                    nn.AdaptiveAvgPool2d(1),
                    View(-1),
                    nn.Linear(n_ch, n_classes)]

        self.network = nn.Sequential(*network)
        print(self)

    def forward(self, x):
        return self.network(x)


class DenseNetBC(nn.Module):
    def __init__(self, depth, growth_rate, efficient=True, ImageNet=False, input_ch=3, n_classes=100):
        super(DenseNetBC, self).__init__()
        """
        Depth is one of [121, 169, 201, 265]. (Please note that DenseNet-264 in the paper is errata. It has to be 265
        including the last fc-layer.)
        """
        init_ch = 2 * growth_rate
        if ImageNet:
            assert depth in [121, 169, 201, 265], "Choose among [121, 169, 201, 265]."
            assert n_classes == 1000, "ImageNet has 1000 classes. Check n_classes: {}.".format(n_classes)

            if depth == 121:
                list_n_layers = [6, 12, 24, 16]
            elif depth == 169:
                list_n_layers = [6, 12, 32, 32]
            elif depth == 201:
                list_n_layers = [6, 12, 48, 32]
            else:
                list_n_layers = [6, 12, 64, 48]

            network = [nn.Conv2d(input_ch, init_ch, 7, stride=2, padding=3, bias=False),
                       nn.BatchNorm2d(init_ch),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2)]

            network += [DenseBlock(list_n_layers[0], init_ch, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = init_ch + growth_rate * list_n_layers[0]

            network += [TransitionLayer(n_ch)]
            network += [DenseBlock(list_n_layers[1], n_ch // 2, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = n_ch // 2 + growth_rate * list_n_layers[1]

            network += [TransitionLayer(n_ch)]
            network += [DenseBlock(list_n_layers[2], n_ch // 2, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = n_ch // 2 + growth_rate * list_n_layers[2]

            network += [TransitionLayer(n_ch)]
            network += [DenseBlock(list_n_layers[3], n_ch // 2, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = n_ch //2 + growth_rate * list_n_layers[3]

        else:
            assert depth in [40, 100, 190, 250]
            n_layers = ((depth - 4) // 3) // 2  # Dividing 2 is because there are two weighted layers in one dense layer
            # in DenseNet BC, i.e. 1x1 and 3x3 convolutions.

            network = [nn.Conv2d(input_ch, init_ch, 3, padding=1, bias=False)]
            network += [DenseBlock(n_layers, init_ch, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = init_ch + growth_rate * n_layers

            network += [TransitionLayer(n_ch)]
            network += [DenseBlock(n_layers, n_ch // 2, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = n_ch // 2 + growth_rate * n_layers

            network += [TransitionLayer(n_ch)]
            network += [DenseBlock(n_layers, n_ch // 2, growth_rate, bottleneck=True, efficient=efficient)]
            n_ch = n_ch // 2 + growth_rate * n_layers

        network += [nn.BatchNorm2d(n_ch),
                    nn.ReLU(True),
                    nn.AdaptiveAvgPool2d(1),
                    View(-1),
                    nn.Linear(n_ch, n_classes)]

        self.network = nn.Sequential(*network)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)

        print(self)
        print("# of params: {}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, x):
        return self.network(x)


class TransitionLayer(nn.Module):
    def __init__(self, n_ch, compress_factor=0.5):
        super(TransitionLayer, self).__init__()
        layer = [nn.BatchNorm2d(n_ch),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(n_ch, int(n_ch * compress_factor), kernel_size=1, bias=False),
                 nn.AvgPool2d(kernel_size=2, stride=2)]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    densenet_bc = DenseNetBC(depth=100, growth_rate=12, n_classes=100, efficient=False)
    flops, params = get_model_complexity_info(densenet_bc, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    print("flops: {}, params: {}".format(flops, params))
