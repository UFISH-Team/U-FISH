import typing as T
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from scipy.ndimage import rotate
from skimage.morphology import dilation
import skimage.morphology as morphology  # noqa: F401


class FISHSpotsDataset(Dataset):
    def __init__(
            self, meta_csv, root_dir,
            spot_footprint='morphology.disk(2)',
            transform=None):
        self.meta_data = pd.read_csv(meta_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.spot_footprint = eval(spot_footprint)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 0])
        image = io.imread(img_path)
        image = np.expand_dims(image, axis=0)

        coord_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 1])
        coordinates = pd.read_csv(coord_path)
        mask = self.coords_to_mask(coordinates, image.shape[1:])

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def coords_to_mask(self, coords: pd.DataFrame, shape: T.Tuple[int, int]):
        mask = np.zeros(shape, dtype=np.uint8)
        for _, row in coords.iterrows():
            y, x = int(row['axis-0']), int(row['axis-1'])
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = 1
        mask = dilation(mask, self.spot_footprint)
        mask = np.expand_dims(mask, axis=0)
        return mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() < self.p:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=2)
        return {'image': image, 'mask': mask}


class RandomRotation:
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        image, mask = sample['image'], sample['mask']
        image = rotate(
            image, angle, axes=(1, 2),
            mode='reflect', order=1, reshape=False)
        mask = rotate(
            mask, angle, axes=(1, 2),
            mode='reflect', order=1, reshape=False)
        return {'image': image, 'mask': mask}


class ToTensorWrapper:
    def __call__(self, sample):
        return {
            'image': torch.tensor(sample['image'], dtype=torch.float32),
            'mask': torch.tensor(sample['mask'], dtype=torch.float32)
        }
