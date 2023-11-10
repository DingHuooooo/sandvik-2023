from PIL import Image
import os
from torchvision import transforms
import torch
import random

class JointTransform:
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    def __call__(self, image, mask):
        rand_seed = random.randint(0, 2**32)  # Generate a new seed for every call
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)  # Ensure image and mask get the same transformation
        image = self.transform(image)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        mask = self.transform(mask)
        return image, mask


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, input_size=1024, need_transform=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.need_transform = need_transform
        self.input_size = input_size
        self.joint_transform = JointTransform(input_size)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

        self.filepaths = self.get_filepaths()
        random.shuffle(self.filepaths)  # Finally, shuffle the whole dataset

    def get_filepaths(self):
        filepaths = [f for f in os.listdir(self.img_dir)]

        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filepaths[idx])
        gt_path = os.path.join(self.gt_dir, self.filepaths[idx].rsplit('.', 1)[0] + "_bitmap.jpg")
    
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')

        if self.need_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask