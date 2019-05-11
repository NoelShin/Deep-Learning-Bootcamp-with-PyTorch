if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from torch.optim.lr_scheduler import MultiStepLR
    from CIFAR10_pipeline import CustomCIFAR10
    from CIFAR10_models import SEResidualNetwork

    BATCH_SIZE = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    dataset = CustomCIFAR10(train=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    # 128 of minibatch size was used for the paper.

    se_resnet = SEResidualNetwork(3).to(device)

    criterion = nn.CrossEntropyLoss()

    optim_se_resnet = torch.optim.SGD(se_resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_se_resnet = MultiStepLR(optim_se_resnet, milestones=[32000, 48000], gamma=0.1)

    iter_total = 0
    while iter_total < 640000:
        for input, label in data_loader:
            iter_total += 1
            input, label = input.to(device), label.to(device)

            output = se_resnet(input)

            loss = criterion(output, label)
            optim_se_resnet.zero_grad()
            loss.backward()
            optim_se_resnet.step()
            scheduler_se_resnet.step()

            n_correct_answers = torch.sum(torch.eq(output.argmax(dim=1), label))

            print("Loss : {:.{prec}}, Acc : {:.{prec}}".format(loss.detach().item(),
                  (float(n_correct_answers.item()) / BATCH_SIZE) * 100, prec=4))

            if iter_total == 640000:
                break
