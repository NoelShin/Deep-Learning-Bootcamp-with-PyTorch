import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    # Set hyper parameters
    BATCH_SIZE = 32
    BETA1 = 0.5
    BETA2 = 0.99
    CLASS_NUMBER = 10
    EPOCHS = 1
    EPSILON = 1e-8
    LR = 2e-4
    MODEL = 'MLP'  # choose among CNN or MLP
    REPORT_FREQ = 100

    # Construct input pipeline
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    # Define model, loss function, optimizer
    if MODEL == 'CNN':
        from MNIST_models import CNN
        model = CNN()
    elif MODEL == 'MLP':
        from MNIST_models import MLP
        model = MLP()
    else:
        raise NotImplementedError("Invalid model type {}. Choose in [CNN, MLP]".format(MODEL))

    cel = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), betas=(BETA1, BETA2), lr=LR, eps=EPSILON)

    # Training loop
    start_time = time.time()

    total_step = 0
    list_loss = list()
    list_acc = list()
    for epoch in range(EPOCHS):
        for i, data in enumerate(data_loader):
            total_step += 1
            input_tensor, label = data[0], data[1]

            input_tensor = input_tensor.view(BATCH_SIZE, -1) if MODEL == 'MLP' else input_tensor

            total_step += 1
            classification_result = model(input_tensor)
            loss = cel(classification_result, label)
            list_loss.append(loss.detach().item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            nb_correct_answer = torch.eq(classification_result.argmax(dim=1), label).sum()
            acc = float(nb_correct_answer.item()) / BATCH_SIZE * 100
            list_acc.append(acc)

            if total_step % REPORT_FREQ == 0:
                print("Epoch {} Acc. is {}%".format(epoch + 1, acc))
    print("Training time taken: {} seconds".format(time.time() - start_time))

    # Save the trained model
    torch.save(model, './basics/classification/MNIST_model_{}.pt'.format(MODEL))

    # Visualize and analyze results
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss, linestyle='--', color='g')
    plt.title('Classification_MNIST_{}'.format(MODEL))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('./basics/classification/MNIST_Loss_graph_{}.png'.format(MODEL))
    plt.show()

    plt.figure()
    plt.plot(range(len(list_acc)), list_acc, linestyle='--', color='b')
    plt.title('Classification_MNIST_{}'.format(MODEL))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.savefig('./basics/classification/MNIST_Accuracy_graph_{}.png'.format(MODEL))
    plt.show()
