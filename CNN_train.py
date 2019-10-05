import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 16x28x28
        self.layer_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)  # 32x14x14
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)  # 64x7x7
        self.layer_3 = nn.AdaptiveAvgPool2d((1, 1))  # 64x1x1
        self.layer_4 = nn.Linear(in_features=64, out_features=10)  # 10
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.act(self.input_layer(x))  # 16x28x28
        x2 = self.act(self.layer_1(x1))  # 32x14x14
        x3 = self.act(self.layer_2(x2))  # 64x7x7
        x4 = self.layer_3(x3)  # Bx64x1x1
        x5 = x4.view(x4.shape[0], 64)  # x4.shape : Bx64x1x1  >> Bx64  *squeeze 1x64x1x1 >> 64
        output = self.layer_4(x5)  # Bx10
        return output


dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNN()

criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=0.01)  # weight_new = weight_old - weight_gradient * lr


list_loss = []
list_acc = []
for epoch in range(1):
    for input, label in tqdm(data_loader):
        # label 32
        output = model(input)  # 32x10
        loss = criterion(output, label)  # 1

        optim.zero_grad()
        loss.backward()
        optim.step()
        list_loss.append(loss.detach().item())

        n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item()
        print("Accuracy: ", n_correct_answers / 32.0 * 100)
        list_acc.append(n_correct_answers / 32.0 * 100)



plt.plot(list_loss)
plt.plot(list_acc)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()