if __name__ == '__main__':
    import os
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    from torch.utils.data import DataLoader
    from models import Discriminator, Generator, weights_init
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from time import time
    from tqdm import tqdm
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    BETA1, BETA2 = 0.5, 0.99
    BATCH_SIZE = 16
    DATASET_NAME = 'MNIST'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    EPOCHS = 1
    ITER_REPORT = 10
    LATENT_DIM = 100
    LR = 2e-4
    N_D_STEP = 1

    if DATASET_NAME == 'CIFAR10':
        from torchvision.datasets import CIFAR10
        transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        dataset = CIFAR10(root='./datasets', train=True, transform=transforms, download=True)
    elif DATASET_NAME == 'LSUN':
        from torchvision.datasets import LSUN
        transforms = Compose([Resize(64), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset = LSUN(root='./datasets/LSUN', classes=['bedroom_train'], transform=transforms)
    elif DATASET_NAME == 'MNIST':
        from torchvision.datasets import MNIST
        transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        dataset = MNIST(root='./datasets', train=True, transform=transforms, download=True)
    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    D = Discriminator(DATASET_NAME).apply(weights_init).to(DEVICE)
    G = Generator(DATASET_NAME).apply(weights_init).to(DEVICE)
    print(D, G)
    criterion = nn.BCELoss()

    optim_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    optim_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))

    list_D_loss = list()
    list_G_loss = list()
    total_step = 0

    st = time()
    for epoch in range(EPOCHS):
        for data in tqdm(data_loader):
            total_step += 1
            real, label = data[0].to(DEVICE), data[1].to(DEVICE)
            z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)

            fake = G(z)

            fake_score = D(fake.detach())
            real_score = D(real)

            D_loss = 0.5 * (criterion(fake_score, torch.zeros_like(fake_score).to(DEVICE))
                            + criterion(real_score, torch.ones_like(real_score).to(DEVICE)))
            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()
            list_D_loss.append(D_loss.detach().cpu().item())

            if total_step % N_D_STEP == 0:
                fake_score = D(fake)
                G_loss = criterion(fake_score, torch.ones_like(fake_score))
                optim_G.zero_grad()
                G_loss.backward()
                optim_G.step()
                list_G_loss.append(G_loss.detach().cpu().item())

                if total_step % ITER_REPORT == 0:
                    print("Epoch: {}, D_loss: {:.{prec}} G_loss: {:.{prec}}"
                          .format(epoch, D_loss.detach().cpu().item(), G_loss.detach().cpu().item(), prec=4))

    torch.save(D.state_dict(), '{}_D.pt'.format(DATASET_NAME))
    torch.save(G.state_dict(), '{}_G.pt'.format(DATASET_NAME))

    plt.figure()
    plt.plot(range(0, len(list_D_loss)), list_D_loss, linestyle='--', color='r', label='Discriminator loss')
    plt.plot(range(0, len(list_G_loss) * N_D_STEP, N_D_STEP), list_G_loss, linestyle='--', color='g',
             label='Generator loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss.png')

    print(time() - st)
