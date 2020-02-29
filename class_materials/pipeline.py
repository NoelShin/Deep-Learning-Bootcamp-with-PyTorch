import os
import random
from random import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, is_train=True):
        dataset_name = "my_dataset"
        dir_datasets = "./datasets"

        if is_train:
            dir_A = os.path.join(dir_datasets, dataset_name, "Train", 'A')
            dir_B = os.path.join(dir_datasets, dataset_name, "Train", 'B')
            self.list_paths_A, self.list_paths_B = sorted(os.listdir(dir_A)), sorted(os.listdir(dir_B))
            self.dir_A, self.dir_B = dir_A, dir_B

        else:
            dir_A = os.path.join(dir_datasets, dataset_name, "Test", 'A')
            self.list_paths_A = sorted(os.listdir(dir_A))
            self.dir_A = dir_A

        self.dataset_name = dataset_name
        self.load_size = 128
        self.is_train = is_train

    def __getitem__(self, index):
        transforms = [Resize((self.load_size, self.load_size), Image.BILINEAR)]
        transforms += [RandomHorizontalFlip()] if random() > 0.5 else []
        transforms += [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        image_A = Image.open(os.path.join(self.dir_A, self.list_paths_A[index]))

        if self.is_train:
            index_random = random.randint(0, len(self.list_paths_B) - 1)
            image_B = Image.open(os.path.join(self.dir_B, self.list_paths_B[index_random]))

        transforms = Compose(transforms)

        A = transforms(image_A)
        B = transforms(image_B) if self.is_train else 0

        return A, B

    def __len__(self):
        return len(self.list_paths_A)




