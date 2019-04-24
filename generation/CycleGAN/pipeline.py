import os
from os.path import join
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
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
        self.load_size = opt.load_size

    def __getitem__(self, index):
        transforms = list()
        if self.dataset_name == 'facades':
            transforms += [Resize((self.load_size, self.load_size), Image.BILINEAR)]
            transforms += [RandomHorizontalFlip()] if random.random() > 0.5 else []
            transforms += [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        else:
            transforms += [Resize((self.load_size, self.load_size), Image.BILINEAR)]
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
    dir_image = './datasets/facades/Train/Input'
    list_paths = os.listdir(dir_image)
    list_sizes = []
    for path in list_paths:
        size = Image.open(os.path.join(dir_image, path)).size
        print(size)
        list_sizes.append(size)

