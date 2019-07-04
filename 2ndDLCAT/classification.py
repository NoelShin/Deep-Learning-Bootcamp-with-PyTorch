# Run this cell to mount your Google Drive.
# from google.colab import drive
# drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


print(torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu:0")
BATCH_SIZE = 64
EPOCHS = 5

transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
dataset = MNIST(root='.',
                download=True,
                transform=transform,
                train=True)

dataloader = DataLoader(dataset=dataset,
                        batch_size=64,
                        shuffle=True)

model = MLP().to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss()

optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

total_step = 0
list_acc = list()
list_loss = list()
for epoch in range(EPOCHS):
    for input, label in dataloader:
        total_step += 1

        input = input.view(input.shape[0], -1).to(DEVICE)
        label = label.to(DEVICE)
        output = model(input)
        loss = criterion(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        estimation = torch.argmax(output, dim=1)
        n_correct_answers = torch.sum(torch.eq(estimation, label))
        acc = (float(n_correct_answers) / BATCH_SIZE) * 100

        list_acc.append(acc)
        list_loss.append(loss.item())

        if total_step % 10 == 0:
            print("Total step: {:d}, Loss: {:.3f}, Acc: {:.3f}"
                  .format(total_step, loss.item(), acc))

torch.save(model.state_dict(), './classification_model.pt')


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])
fig.suptitle("Loss & Accuracy graph")

axs[0].plot(range(total_step), list_loss, linestyle='--', label='Loss')
axs[0].set_ylabel("Loss")
axs[0].set_xlabel("Iteration")
axs[0].grid(True)

axs[1].plot(range(total_step), list_acc, linestyle='--', label='Accuracy')
axs[1].set_ylabel("Accuracy(%)")
axs[1].set_xlabel("Iteration")
axs[1].grid(True)

plt.show()

# Below is for testing model.

# transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
# dataset = MNIST(root='.', train=False, transform=transform)
# data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
#
# model = MLP()
# model.load_state_dict(torch.load('./classification_model.pt'))
#
# total_correct_answers = 0
# with torch.no_grad():
#   for input, label in data_loader:
#     input = input.view(input.shape[0], -1)
#     output = model(input)
#     estimation = torch.argmax(output, dim=1)
#     n_correct_answers = torch.sum(torch.eq(estimation, label))
#     total_correct_answers += n_correct_answers
#
# acc = float(total_correct_answers) / len(dataset) * 100
# print("Acc: {:.3f}%".format(acc))
