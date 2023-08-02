import typing as T
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage as ndi
from skimage.io import imread
from skimage.morphology import dilation
import skimage.morphology as morphology  # noqa: F401

from ..utils.misc import scale_image


class Reader:
    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass


class FileReader(Reader):
    """Read images and coordinates from
    meta_csv and files."""
    def __init__(self, root_dir: str, meta_csv_path: str):
        self.root_dir = root_dir
        self.meta_data = pd.read_csv(meta_csv_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 0])
        image = imread(img_path)
        coord_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 1])
        coordinates = pd.read_csv(coord_path)
        sample = {'image': image, 'coords': coordinates.values}
        return sample


class ListReader(Reader):
    """Read images and coordinates from
    a list of images and coordinates."""
    def __init__(
            self,
            img_list: T.List[np.ndarray],
            coord_list: T.List[np.ndarray]):
        self.img_list = img_list
        self.coord_list = coord_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int):
        image = self.img_list[idx]
        coordinates = self.coord_list[idx]
        sample = {'image': image, 'coords': coordinates}
        return sample


class FISHSpotsDataset(Dataset):
    def __init__(
            self, reader: Reader,
            process_func: T.Optional[T.Callable] = None,
            transform=None):
        """FISH spots dataset.

        Args:
            reader: The reader to read images and coordinates.
            process_func: The function to process the target image.
            transform: The transform to apply to the samples.
        """
        self.reader = reader
        self.transform = transform
        self.process_func = process_func or self.gaussian_filter

    @staticmethod
    def gaussian_filter(mask: np.ndarray, sigma=1) -> np.ndarray:
        """Apply Gaussian filter to the mask."""
        peak = np.stack(np.where(mask > 0), axis=1)
        res = ndi.gaussian_filter(mask, sigma=sigma)
        peak_val = res[tuple(peak.T)]
        res /= peak_val.min()
        return res

    @staticmethod
    def dialate_mask(
            mask: np.ndarray,
            footprint: np.ndarray = morphology.disk(2)
            ) -> np.ndarray:
        """Dialate the mask."""
        return dilation(mask, footprint=footprint)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int):
        data = self.reader[idx]
        image, coords = data['image'], data['coords']
        target = self.coords_to_target(coords, image.shape)
        image = scale_image(image)
        image = np.expand_dims(image, axis=0)
        sample = {'image': image, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def coords_to_target(
            self, coords: np.ndarray,
            shape: T.Tuple[int, int],
            ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        # remove out-of-bound coordinates
        c = coords
        c = (c + 0.5).astype(np.uint32)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < shape[0])]
        c = c[(c[:, 1] >= 0) & (c[:, 1] < shape[1])]
        mask[c[:, 0], c[:, 1]] = 1
        mask = self.process_func(mask)
        mask = np.expand_dims(mask, axis=0)
        return mask

    @classmethod
    def from_meta_csv(
            cls,
            root_dir: str,
            meta_csv_path: str,
            process_func: T.Optional[T.Callable] = None,
            transform=None):
        """Create a dataset from a meta CSV file."""
        reader = FileReader(root_dir, meta_csv_path)
        return cls(reader, process_func, transform)

    @classmethod
    def from_list(
            cls,
            img_list: T.List[np.ndarray],
            coord_list: T.List[np.ndarray],
            process_func: T.Optional[T.Callable] = None,
            transform=None):
        """Create a dataset from a list of images and coordinates."""
        reader = ListReader(img_list, coord_list)
        return cls(reader, process_func, transform)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if random.random() < self.p:
            image = np.flip(image, axis=2)
            target = np.flip(target, axis=2)
        return {'image': image, 'target': target}


class RandomRotation:
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        image, target = sample['image'], sample['target']
        image = ndi.rotate(
            image, angle, axes=(1, 2),
            mode='reflect', order=1, reshape=False)
        target = ndi.rotate(
            target, angle, axes=(1, 2),
            mode='reflect', order=1, reshape=False)
        return {'image': image, 'target': target}


class ToTensorWrapper:
    def __call__(self, sample):
        return {
            'image': torch.tensor(sample['image'], dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32)
        }


composed_transform = Compose([
    RandomHorizontalFlip(),
    RandomRotation(),
    ToTensorWrapper(),
])
