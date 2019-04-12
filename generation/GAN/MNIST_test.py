if __name__ == '__main__':
    import os
    from os.path import join
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image
    from MNIST_models import Generator
    import time

    st = time.time()

    BATCH_SIZE = 16
    IMAGE_DIR = './GAN/checkpoints/MNIST/Image/Test'
    LATENT_DIM = 100
    MODEL_DIR = './GAN/checkpoints/MNIST/Model'
    os.makedirs(IMAGE_DIR, exist_ok=True)

    dataset = MNIST(root='./datasets', train=False, transform=ToTensor(), download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=0, shuffle=False)

    G = Generator()
    G.load_state_dict(torch.load(join(MODEL_DIR, 'Latest_G.pt')).state_dict())

    for p in G.parameters():
        p.requires_grad_(False) if p.requires_grad else None

    z = torch.randn(BATCH_SIZE, LATENT_DIM)

    fake = G(z).view(BATCH_SIZE, 1, 28, 28)
    save_image(fake.detach(), join(IMAGE_DIR, 'Latest_G_results.png'), nrow=4, normalize=True)

    print("Total time taken: ", time.time() - st)
