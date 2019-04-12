if __name__ == '__main__':
    import os
    from os.path import join
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.datasets import MNIST
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from models import Discriminator, Generator, weights_init
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from time import time
    from tqdm import tqdm  # For visualizing a time bar for training

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.device_count() > 1 else ''

    BATCH_SIZE = 16
    BETA1, BETA2 = 0.5, 0.99
    CNN = False
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    DIR_ANALYSIS = './SGAN/checkpoints/Analysis'
    DIR_IMAGE = './SGAN/checkpoints/Image/Training'
    DIR_MODEL = './SGAN/checkpoints/Model'
    EPOCHS = 25
    ITER_DISPLAY = 100
    ITER_REPORT = 10
    LATENT_DIM = 100
    LR = 2e-4
    N_D_STEP = 1

    os.makedirs(DIR_ANALYSIS, exist_ok=True)
    os.makedirs(DIR_IMAGE, exist_ok=True)
    os.makedirs(DIR_MODEL, exist_ok=True)

    transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='./datasets', train=True, transform=transforms, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    D = Discriminator(cnn=CNN).apply(weights_init).to(DEVICE)
    G = Generator(cnn=CNN).apply(weights_init).to(DEVICE)
    print(D, G)

    CELoss = nn.CrossEntropyLoss()
    BCELoss = nn.BCELoss()

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
            real = real.view(real.shape[0], -1) if not CNN else real
            z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake = G(z)

            class_fake, validity_fake = D(fake.detach())
            class_real, validity_real = D(real)
            loss_class_fake = CELoss(class_fake, label)
            loss_class_real = CELoss(class_real, label)
            loss_fake = BCELoss(validity_fake, torch.zeros_like(validity_fake).to(DEVICE))
            loss_real = BCELoss(validity_real, torch.ones_like(validity_real).to(DEVICE))
            D_loss = (loss_class_fake + loss_class_real + loss_fake + loss_real).mean()

            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()

            list_D_loss.append(D_loss.detach().cpu().item())

            if total_step % N_D_STEP == 0:
                class_fake, validity_fake = D(fake)
                loss_class_fake = CELoss(class_fake, label)
                loss_fake = BCELoss(validity_fake, torch.ones_like(validity_fake))

                G_loss = (loss_class_fake + loss_fake).mean()

                optim_G.zero_grad()
                G_loss.backward()
                optim_G.step()

                list_G_loss.append(G_loss.detach().cpu().item())

                if total_step % ITER_REPORT == 0:
                    print(" Epoch: {}, D_loss: {:.{prec}}, G_loss {:.{prec}}"
                          .format(epoch + 1, D_loss.detach().cpu().item(), G_loss.detach().cpu().item(), prec=4))

                if total_step % ITER_DISPLAY == 0:
                    save_image(fake.detach().view(BATCH_SIZE, 1, 28, 28),
                               join(DIR_IMAGE, '{}_fake.png'.format(epoch + 1)), nrow=int(BATCH_SIZE ** (1/2)),
                               normalize=True)
                    save_image(real.view(BATCH_SIZE, 1, 28, 28), join(DIR_IMAGE, '{}_real.png'.format(epoch + 1)),
                               nrow=int(BATCH_SIZE ** (1/2)), normalize=True)

    print("Total time taken: ", time() - st)

    torch.save(G.state_dict(), join(DIR_MODEL, 'G.pt'))
    torch.save(D.state_dict(), join(DIR_MODEL, 'D.pt'))

    plt.figure()
    plt.title('Semi-Supervised GAN Training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(0, len(list_D_loss)), list_D_loss, linestyle='--', color='r', label='Discriminator loss')
    plt.plot(range(0, len(list_G_loss) * N_D_STEP, N_D_STEP), list_G_loss, linestyle='--', color='g',
             label='Generator loss')
    plt.legend()
    plt.savefig(join(DIR_ANALYSIS, 'SGAN_training.png'))

