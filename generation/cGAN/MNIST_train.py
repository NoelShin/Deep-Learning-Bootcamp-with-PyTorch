from os.path import join
import torch.nn as nn


if __name__ == '__main__':
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor
    from torchvision.utils import save_image
    from MNIST_models import Generator, Discriminator
    import datetime

    BATCH_SIZE = 16  # Original batch size is 100.
    EPOCHS = 25
    IMAGE_DIR = './cGAN/checkpoints/MNIST/Image/Training'
    IMAGE_SIZE = 28
    ITER_DISPLAY = 100
    ITER_REPORT = 10
    LATENT_DIM = 100  # Original latent dimension is 100.
    MODEL_DIR = './cGAN/checkpoints/MNIST/Model'
    OUT_CHANNEL = 1
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    transform = Compose([ToTensor()])  # The output activation in the generator is sigmoid. Thus we don't normalize
    # input to have [-1, 1] but [0, 1]. ToTensor() does it.
    dataset = MNIST(root='./datasets', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # They don't refer about parameter initialization.
    D = Discriminator()
    G = Generator()
    print(D)
    print(G)

    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))

    st = datetime.datetime.now()
    iter = 0
    for epoch in range(EPOCHS):
        for i, data in enumerate(data_loader):
            iter += 1

            real, label = data[0].view(BATCH_SIZE, -1), data[1]
            one_hot = torch.zeros(BATCH_SIZE, 10)
            one_hot.scatter_(dim=1, index=label.view(BATCH_SIZE, 1), src=torch.ones(BATCH_SIZE, 1))

            z = torch.rand(BATCH_SIZE, LATENT_DIM)
            fake = G(z, one_hot)

            loss_D = -torch.mean(torch.log(D(real, one_hot)) + torch.log(1 - D(fake.detach(), one_hot)))
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            loss_G = -torch.mean(torch.log(D(fake, one_hot)))  # Non saturating loss
            # For saturaing loss, loss_G = torch.mean(torch.log(1-D(fake)))
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if iter % ITER_DISPLAY == 0:
                fake = fake.view(BATCH_SIZE, OUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
                real = real.view(BATCH_SIZE, OUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
                save_image(fake, IMAGE_DIR + '/{}_fake.png'.format(epoch + 1), nrow=4, normalize=True)
                save_image(real, IMAGE_DIR + '/{}_real.png'.format(epoch + 1), nrow=4, normalize=True)

            if iter % ITER_REPORT == 0:
                print("Epoch: {} Iter: {} Loss D: {:.{prec}} Loss G: {:.{prec}}"
                      .format(epoch + 1, iter, loss_D.detach().item(), loss_G.detach().item(), prec=4))

    torch.save(D, join(MODEL_DIR, 'Latest_D.pt'))
    torch.save(G, join(MODEL_DIR, 'Latest_G.pt'))
    print("Total time taken: ", datetime.datetime.now() - st)
