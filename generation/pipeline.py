import os
from os.path import join
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        dataset_name = opt.dataset_name
        dir_datasets = opt.dir_datasets
        is_train = opt.is_train

        if is_train:
            dir_input = join(dir_datasets, dataset_name, 'Train', 'Input')
            dir_real = join(dir_datasets, dataset_name, 'Train', 'Real')
            self.list_paths_input, self.list_paths_real = sorted(os.listdir(dir_input)), sorted(os.listdir(dir_real))
            self.dir_input, self.dir_real = dir_input, dir_real
        else:
            dir_input = join(dir_datasets, dataset_name, 'Test', 'Input')
            self.list_paths_input = sorted(os.listdir(dir_input))
            self.dir_input, self.dir_real = dir_input, None

        self.dataset_name = opt.dataset_name
        self.is_train = is_train

    def __getitem__(self, index):
        transforms = list()
        if self.dataset_name == 'facades':
            transforms += [RandomHorizontalFlip()] if random.random() > 0.5 else []
            transforms += [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        else:
            transforms += [ToTensor(), Normalize(mean=[0.5], std=[0.5])]
        transforms = Compose(transforms)

        image_input = Image.open(join(self.dir_input, self.list_paths_input[index])).convert('RGB')
        input = transforms(image_input)

        if self.is_train:
            image_real = Image.open(join(self.dir_real, self.list_paths_real[random.randint(0, len(self.list_paths_real)
                                                                                            - 1)])).convert('RGB')
            real = transforms(image_real)
        else:
            real = 0

        return input, real

    def __len__(self):
        return len(self.list_paths_input)


if __name__ == '__main__':
    from options import TrainOption
    from torch.utils.data import DataLoader
    import numpy as np
    from PIL import Image
    opt = TrainOption().parse()
    dataset = CustomDataset(opt)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for input, real in dataloader:
        input = np.array(input[0])
        real = np.array(real[0])

        input -= -1.
        input *= 127.5
        input = input.astype(np.uint8)
        Image.fromarray(input.transpose(1, 2, 0), mode='RGB').show()

        real -= -1.0
        real *= 127.5
        real = real.astype(np.uint8)
        Image.fromarray(real.transpose(1, 2, 0), mode='RGB').show()

        break
