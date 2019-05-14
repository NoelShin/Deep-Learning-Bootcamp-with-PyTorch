if __name__ == '__main__':
    import os
    from os.path import join
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from options import TrainOption
    from pipeline import CustomDataset
    from networks import Discriminator, Generator, update_lr, weights_init
    from utils import ImageBuffer
    from datetime import datetime

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    batch_size = opt.batch_size
    beta_1, beta_2 = opt.beta_1, opt.beta_2
    debug = opt.debug
    dir_image_train = opt.dir_image_train
    dir_model = opt.dir_model
    epoch_decay = opt.epoch_decay
    epoch_save = opt.epoch_save
    iter_display = opt.iter_display
    iter_report = opt.iter_report
    iter_val = opt.iter_val
    n_epochs = opt.n_epochs
    lambda_cycle = opt.lambda_cycle
    lr = opt.lr
    n_buffer_images = opt.n_buffer_images
    num_workers = opt.n_workers
    val_during_training = opt.val_during_training
    if val_during_training:
        from options import TestOption
        opt_test = TestOption().parse()
        dir_image_test = opt.dir_image_test

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    dataset = CustomDataset(opt, 'train')
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    D_A = Discriminator(opt).apply(weights_init).to(device)
    D_B = Discriminator(opt).apply(weights_init).to(device)
    G_A = Generator(opt).apply(weights_init).to(device)
    G_B = Generator(opt).apply(weights_init).to(device)
    print(D_A)
    print(G_A)

    optim_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(beta_1, beta_2))
    optim_G = torch.optim.Adam(list(G_A.parameters()) + list(G_B.parameters()), lr=lr, betas=(beta_1, beta_2))

    loss_GAN = nn.MSELoss()  # Note that the authors used LSGAN.
    loss_cycle = nn.L1Loss()  # Cycle consistency loss.

    image_buffer_A = ImageBuffer(n_images=n_buffer_images)
    image_buffer_B = ImageBuffer(n_images=n_buffer_images)

    st = datetime.now()
    iter_total = 0
    for epoch in range(n_epochs):
        if (epoch + 1) > epoch_decay:
            lr = update_lr(opt.lr, lr, n_epochs - epoch_decay, optim_D, optim_G)
        for A, B in data_loader:
            iter_total += 1
            A, B = A.to(device), B.to(device)

            fake_B = G_A(A)
            fake_A = G_B(B)

            val_fake_B = D_B(image_buffer_B(fake_B.detach()))
            val_real_B = D_B(B)
            val_fake_A = D_A(image_buffer_A(fake_A.detach()))
            val_real_A = D_A(A)

            loss_D = 0
            loss_D += loss_GAN(val_fake_B, torch.zeros_like(val_fake_B).to(device))
            loss_D += loss_GAN(val_fake_A, torch.zeros_like(val_fake_A).to(device))
            loss_D += loss_GAN(val_real_B, torch.ones_like(val_real_B).to(device))
            loss_D += loss_GAN(val_real_A, torch.ones_like(val_real_A).to(device))
            loss_D *= 0.5

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            val_fake_B = D_B(fake_B)
            val_fake_A = D_A(fake_A)
            cycle_A = G_B(fake_B)
            cycle_B = G_A(fake_A)

            loss_G = 0
            loss_G += loss_GAN(val_fake_B, torch.ones_like(val_fake_B).to(device))
            loss_G += loss_GAN(val_fake_A, torch.ones_like(val_fake_A).to(device))
            loss_G += lambda_cycle * (loss_cycle(cycle_A, A) + loss_cycle(cycle_B, B))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if iter_total % iter_report == 0:
                print("Epoch: {}, Iter: {}, Loss D: {:.{prec}}, Loss G: {:.{prec}}"
                      .format(epoch + 1, iter_total, loss_D.detach().item(), loss_G.detach().item(), prec=4))

            if iter_total % iter_display == 0:
                save_image(fake_A.detach(), join(dir_image_train, '{}_fake_A.png'.format(epoch + 1)), nrow=1,
                           normalize=True)
                save_image(fake_B.detach(), join(dir_image_train, '{}_fake_B.png'.format(epoch + 1)), nrow=1,
                           normalize=True)

            if (iter_total % iter_val == 0) and val_during_training:
                dir_image_test_A2B = join(dir_image_test, str(iter_total), 'A2B')
                dir_image_test_B2A = join(dir_image_test, str(iter_total), 'B2A')
                os.makedirs(dir_image_test_A2B, exist_ok=True)
                os.makedirs(dir_image_test_B2A, exist_ok=True)

                # Generate fake B images with input A images.
                dataset_test_A2B = CustomDataset(opt_test, 'val_A2B')
                data_loader_test_A2B = DataLoader(dataset=dataset_test_A2B, batch_size=batch_size,
                                                  num_workers=num_workers, shuffle=False)

                for p in G_A.parameters():
                    p.requires_grad_(False)

                for i, A_val, _ in enumerate(data_loader_test_A2B):
                    A_val = A_val.to(device)

                    fake_B = G_A(A_val)

                    save_image(fake_B.detach(), join(dir_image_test_A2B,
                                                     '{}_fake_B.png'.format(i)), nrow=1, normalize=True)

                for p in G_A.parameters():
                    p.requires_grad_(True)

                # Generate fake A image with input B images.
                dataset_test_B2A = CustomDataset(opt_test, 'val_B2A')
                data_loader_test_B2A = DataLoader(dataset=dataset_test_B2A, batch_size=batch_size,
                                                  num_workers=num_workers, shuffle=False)

                for p in G_B.parameters():
                    p.requires_grad_(False)

                for i, B_val, _ in enumerate(data_loader_test_B2A):
                    B_val = B_val.to(device)

                    fake_A = G_B(B_val)

                    save_image(fake_A.detach(), join(dir_image_test_B2A,
                                                     '{}_fake_A.png'.format(i)), nrow=1, normalize=True)

                for p in G_B.parameters():
                    p.requires_grad_(True)

        if (epoch + 1) % epoch_save == 0:
            torch.save(G_A.state_dict(), join(dir_model, '{}_G_A.pt'.format(epoch + 1)))
            torch.save(G_B.state_dict(), join(dir_model, '{}_G_B.pt'.format(epoch + 1)))
            torch.save(D_A.state_dict(), join(dir_model, '{}_D_A.pt'.format(epoch + 1)))
            torch.save(D_B.state_dict(), join(dir_model, '{}_D_B.pt'.format(epoch + 1)))

    print(datetime.now() - st)
