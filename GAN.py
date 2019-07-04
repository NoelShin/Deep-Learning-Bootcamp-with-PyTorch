# Run this cell to mount your Google Drive.
# from google.colab import drive
# drive.mount('/content/drive')

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')


# Model definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [nn.Linear(in_features=100, out_features=128),
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5)]
        model += [nn.Linear(in_features=128, out_features=256),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5)]
        model += [nn.Linear(in_features=256, out_features=28 * 28),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [nn.Linear(28 * 28, 256),
                 nn.ReLU(inplace=True)]
        model += [nn.Linear(256, 128),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5)]
        model += [nn.Linear(128, 1),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


dataset = MNIST(root='.',
                transform=ToTensor(),
                download=True,
                train=True)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=True,
                                          batch_size=1)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

print(G)
print(D)

G_optim = torch.optim.Adam(params=G.parameters(), lr=0.0002, betas=(0.5, 0.9))
D_optim = torch.optim.Adam(params=D.parameters(), lr=0.0002, betas=(0.5, 0.9))

total_step = 0
for epoch in range(10):
    for real, _ in tqdm(data_loader):
        total_step += 1

        z = torch.rand(real.shape[0], 100).to(DEVICE)

        fake = G(z)
        real = real.view(real.shape[0], -1).to(DEVICE)

        fake_score = D(fake.detach())
        real_score = D(real)

        D_loss = -torch.mean(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8))
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        fake_score = D(fake)

        G_loss = -torch.mean(torch.log(fake_score + 1e-8))
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        if total_step % 100 == 0:
            save_image(fake.view(fake.shape[0], 1, 28, 28),
                       '{}_fake.png'.format(epoch + 1),
                       nrow=1,
                       normalize=True, range=(0, 1))

            save_image(real.view(real.shape[0], 1, 28, 28),
                       '{}_real.png'.format(epoch + 1),
                       nrow=1,
                       normalize=True, range=(0, 1))

torch.save(G.state_dict(), 'G.pt')

# Below is for testing model.
# G = Generator()
# G.load_state_dict(torch.load('./G.pt'))
#
# z = torch.rand(16, 100)
# fake = G(z)
# save_image(fake.view(fake.shape[0], 1, 28, 28),
#            "./test_fake.png",
#            nrow=4,
#            normalize=True,
#            range=(0, 1))


