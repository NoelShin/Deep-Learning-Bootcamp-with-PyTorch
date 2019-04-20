import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Grayscale, ToTensor, Normalize
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, crop_size=0, flip=False):
        super(CustomDataset, self).__init__()
        self.crop_size= crop_size
        self.flip = flip
        self.list_paths = os.listdir(root)
        self.root = root

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.list_paths[index]))  # Open image from the given path.

        # Get transform list
        list_transforms = list()
        if self.crop_size > 0:
            list_transforms.append(RandomCrop((self.crop_size, self.crop_size)))
        if self.flip:
            coin = random.random() > 0.5
            if coin:
                list_transforms.append(RandomHorizontalFlip())
        transforms = Compose(list_transforms)

        image = transforms(image)  # Implement common transform

        input_image = Grayscale(num_output_channels=1)(image)  # For input image, we need to make it B/W.

        input_tensor, target_tensor = ToTensor()(input_image), ToTensor()(image)  # Make input, target as torch.Tensor

        input_tensor = Normalize(mean=[0.5], std=[0.5])(input_tensor)  # As the input tensor has only one channel,
        # Normalize parameters also have one value each.
        target_tensor = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(target_tensor)  # As the target tensor has
        # three channels Normalize parameters also have three values each.

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.list_paths)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np

    dataset = CustomDataset('./datasets/Noel', crop_size=128, flip=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for input, target in dataloader:
        input_image_np = np.array(input[0, 0])
        input_image_np -= input_image_np.min()
        input_image_np /= input_image_np.max()
        input_image_np *= 255.0
        input_image_np = input_image_np.astype(np.uint8)

        input_image = Image.fromarray(input_image_np, mode='L')
        input_image.show()

        target_image_np = np.array(target[0])
        target_image_np -= target_image_np.min()
        target_image_np /= target_image_np.max()
        target_image_np *= 255.0
        target_image_np = target_image_np.astype(np.uint8)

        target_image = Image.fromarray(target_image_np.transpose(1, 2, 0), mode='RGB')
        target_image.show()
        break
