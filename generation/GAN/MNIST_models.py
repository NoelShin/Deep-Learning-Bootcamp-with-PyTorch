import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [nn.Linear(in_features=100, out_features=128), nn.ReLU(inplace=True)]
        model += [nn.Linear(in_features=128, out_features=256), nn.ReLU(inplace=True)]
        model += [nn.Linear(in_features=256, out_features=28 * 28), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

        # "The generator nets used a mixture of rectifier linear activations and sigmoid activations, while the
        #  discriminator net used maxout activations." - Generative Adversarial Networks

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [Maxout(28 * 28, 256, dropout=False, k=5)]
        model += [Maxout(256, 128, dropout=True, k=5)]
        model += [nn.Linear(128, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Maxout(nn.Module):
    def __init__(self, in_features, out_features, k=2, dropout=True, p=0.5):
        super(Maxout, self).__init__()
        model = [nn.Dropout(p)] if dropout else []
        model += [nn.Linear(in_features, out_features * k)]

        self.model = nn.Sequential(*model)
        self.k = k

        # Note that dropout is used before weight multiplication following 'Maxout Networks' paper.
        # "When training with dropout, we perform the elementwise multiplication with the dropout mask immediately prior
        #  to the multiplication by the weights in all cases-we do not drop inputs to the max operator." - Maxout
        #  Networks

    def forward(self, x):
        x = self.model(x)
        x, _ = x.view(x.shape[0], x.shape[1] // self.k, self.k).max(-1)
        return x

