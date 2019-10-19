import os
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from residual_network import ResidualNetwork


dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ResidualNetwork()

criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=0.001)  # weight_new = weight_old - weight_gradient * lr

# if os.path.isfile("./weight_dict.pt"):
#     model_dict = torch.load('./weight_dict.pt')['model_weight']
#     model.load_state_dict(model_dict)
#     adam_dict = torch.load('./weight_dict.pt')['adam_weight']
#     optim.load_state_dict(adam_dict)

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

weight_dict = {'model_weight': model.state_dict(), 'adam_weight': optim.state_dict()}
torch.save(weight_dict, "./weight_dict.pt")
# torch.save(model.state_dict(), "./CNN_model.pt")
# torch.save(optim.state_dict(), "./adam.pt")

plt.plot(list_loss)
plt.plot(list_acc)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()