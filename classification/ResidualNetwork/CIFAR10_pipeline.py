import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Lambda, ToTensor
from PIL import Image
import PIL


class CustomCIFAR10(Dataset):
    def __init__(self, train=True):
        super(CustomCIFAR10, self).__init__()
        self.cifar10_train = CIFAR10(root='./datasets', train=train, download=True)

        # Get only images (i.e. without labels)
        tensors = list()
        for i in range(len(self.cifar10_train)):
            tensors.append(np.array(self.cifar10_train[i][0]))  # Need to convert PIL.Image.Image to numpy.ndarray
        self.per_pixel_mean_grid = np.mean(tensors, axis=0).astype(np.float32)
        # Calculate per-pixel mean along the batch dimension

        if not train:
            self.cifar10_test = CIFAR10(root='./datasets', train=train, download=True)

        self.train = train

    def __getitem__(self, index):
        transforms = list()
        transforms.append(Lambda(self.__to_numpy))  # First convert PIL.Image.Image to numpy.ndarray. HxWxC
        transforms.append(Lambda(self.__per_pixel_subtraction_normalization))  # Subtract per-pixel mean

        if self.train:
            if random.random() > 0.5:
                transforms.append(Lambda(self.__horizontal_flip))  # Flip horizontally with 50:50 chance
            transforms.append(Lambda(self.__pad_and_random_crop))  # Pad 0 along H and W dims and randomly crop

        transforms.append(ToTensor())  # convert numpy.ndarray to torch.tensor (notice that HxWxC -> CxHxW)
        transforms = Compose(transforms)

        if self.train:
            return transforms(self.cifar10_train[index][0]), self.cifar10_train[index][1]
        else:
            return transforms(self.cifar10_test[index][0]), self.cifar10_test[index][1]

    def __len__(self):
        if self.train:
            return len(self.cifar10_train)
        else:
            return len(self.cifar10_test)

    def __per_pixel_subtraction_normalization(self, x):
        return (x - self.per_pixel_mean_grid) / 255.

    @staticmethod
    def __pad_and_random_crop(x):
        p = 4  # Original paper pad 4 on each side
        x = np.pad(x, ((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)  # Pad along H, W, and C dims
        y_index = random.randint(0, 2 * p - 1)
        x_index = random.randint(0, 2 * p - 1)
        x = x[y_index: y_index + 32, x_index: x_index + 32, :]  # Crop
        return x

    @staticmethod
    def __horizontal_flip(x):
        return np.fliplr(x)

    @staticmethod
    def __to_numpy(x):
        assert isinstance(x, PIL.Image.Image)
        return np.array(x).astype(np.float32)
