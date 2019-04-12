if __name__ == '__main__':
    import os
    from os.path import join
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Normalize, ToTensor
    from torchvision.utils import save_image
    from CIFAR10_models import Generator
    import time

    st = time.time()

    BATCH_SIZE = 16
    IMAGE_DIR = './GAN/checkpoints/CIFAR10/Image/Test'
    IMAGE_SIZE = 32
    LATENT_DIM = 100
    MODEL_DIR = './GAN/checkpoints/CIFAR10/Model'
    OUTPUT_CHANNEL = 3
    os.makedirs(IMAGE_DIR, exist_ok=True)

    transforms = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset = CIFAR10(root='./datasets', train=False, transform=transforms, download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    G = Generator()
    G.load_state_dict(torch.load(join(MODEL_DIR, 'Latest_G.pt')).state_dict())

    for p in G.parameters():
        p.requires_grad_(False) if p.requires_grad else None

    z = torch.randn(BATCH_SIZE, LATENT_DIM)

    fake = G(z).view(BATCH_SIZE, OUTPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    save_image(fake.detach(), join(IMAGE_DIR, 'Latest_G_results.png'), nrow=4, normalize=True)

    print("Total time taken: ", time.time() - st)
