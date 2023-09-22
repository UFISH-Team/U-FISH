import os
import random
from pathlib import Path
import typing as T

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndi
from skimage.io import imread
from skimage.morphology import dilation
import skimage.morphology as morphology  # noqa: F401

from .utils.img import scale_image
from .utils.log import logger


class Reader:
    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass

    def read_coords(self, path: str, ndim: int) -> np.ndarray:
        coordinates = pd.read_csv(path)
        axes = [f'axis-{i}' for i in range(ndim)]
        coords = coordinates[axes].values
        return coords


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
        coords = self.read_coords(coord_path, image.ndim)
        sample = {'image': image, 'coords': coords}
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


class DirReader(Reader):
    """Read images and coordinates from
    a directory of images and coordinates."""
    def __init__(
            self,
            img_dir: str,
            coord_dir: str,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            ):
        self.img_dir = Path(img_dir)
        self.coord_dir = Path(coord_dir)
        self.img_paths = sorted(self.img_dir.glob(img_glob))
        self.coord_paths = sorted(self.coord_dir.glob(coord_glob))
        assert len(self.img_paths) == len(self.coord_paths), \
            "Number of images and coordinates must match."
        self.check_prefix()

    def check_prefix(self):
        for img_path, coord_path in zip(self.img_paths, self.coord_paths):
            img_prefix = img_path.stem
            coord_prefix = coord_path.stem
            if img_prefix != coord_prefix:
                logger.warning(
                    f"Image prefix {img_prefix} does not match "
                    f"coordinate prefix {coord_prefix}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        coord_path = self.coord_paths[idx]
        image = imread(img_path)
        coords = self.read_coords(coord_path, image.ndim)
        sample = {'image': image, 'coords': coords}
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
        self.process_func = process_func

    @staticmethod
    def gaussian_filter(mask: np.ndarray, sigma=1) -> np.ndarray:
        """Apply Gaussian filter to the mask."""
        if mask.max() == 0:
            return mask
        peak = np.stack(np.where(mask > 0), axis=1)
        res = ndi.gaussian_filter(mask, sigma=sigma)
        peak_val = res[tuple(peak.T)]
        res /= peak_val.min()
        return res

    @staticmethod
    def dialate_mask(
            mask: np.ndarray,
            footprint: str = 'disk(2)'
            ) -> np.ndarray:
        """Dialate the mask."""
        _footprint = eval(f"morphology.{footprint}")
        return dilation(mask, footprint=_footprint)

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
        mask = np.zeros(shape, dtype=np.float32)
        # remove out-of-bound coordinates
        c = coords
        c = (c + 0.5).astype(np.uint32)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < shape[0])]
        c = c[(c[:, 1] >= 0) & (c[:, 1] < shape[1])]
        mask[c[:, 0], c[:, 1]] = 1
        if self.process_func:
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

    @classmethod
    def from_dir(
            cls,
            img_dir: str,
            coord_dir: str,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            process_func: T.Optional[T.Callable] = None,
            transform=None):
        """Create a dataset from a directory of images and coordinates."""
        reader = DirReader(img_dir, coord_dir, img_glob, coord_glob)
        return cls(reader, process_func, transform)


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if random.random() < self.p:
            image = np.flip(image, axis=2).copy()
            target = np.flip(target, axis=2).copy()
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            target = np.flip(target, axis=1).copy()
        return {'image': image, 'target': target}


class RandomRotation:
    def __init__(self, p=0.5, angle_range=(-90, 90)):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        image, target = sample['image'], sample['target']
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        image = ndi.rotate(
            image, angle, axes=(1, 2), order=1, reshape=False)
        target = ndi.rotate(
            target, angle, axes=(1, 2), order=1, reshape=False)
        return {'image': image, 'target': target}


class RandomTranslation:
    def __init__(self, p=0.5, shift_range=(-256, 256)):
        self.p = p
        self.shift_range = shift_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        image, target = sample['image'], sample['target']
        shift_y = random.uniform(self.shift_range[0], self.shift_range[1])
        shift_x = random.uniform(self.shift_range[0], self.shift_range[1])
        image = ndi.shift(
            image, (0, shift_y, shift_x), cval=0.0)
        target = ndi.shift(
            target, (0, shift_y, shift_x), cval=0.0)
        return {'image': image, 'target': target}


class GaussianNoise:
    def __init__(self, p=0.5, sigma_range=(0, 0.5)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        image, target = sample['image'], sample['target']
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.random.normal(
            0, sigma, image.shape).astype(np.float32)
        image += noise
        image = np.clip(image, 0, 255)
        return {'image': image, 'target': target}


class SaltAndPepperNoise:
    def __init__(
            self, p=0.5,
            salt_range=(0, 1e-4),
            pepper_range=(0, 1e-4)):
        self.p = p
        self.salt_range = salt_range
        self.pepper_range = pepper_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        p_salt = random.uniform(
            self.salt_range[0], self.salt_range[1])
        p_pepper = random.uniform(
            self.pepper_range[0], self.pepper_range[1])
        image, target = sample['image'], sample['target']
        mask = np.random.choice(
            [0, 1, 2], size=image.shape,
            p=[(1-p_salt-p_pepper), p_salt, p_pepper])
        image[mask == 1] = 0.0
        image[mask == 2] = 255.0
        return {'image': image, 'target': target}


class ToTensorWrapper:
    def __call__(self, sample):
        return {
            'image': torch.tensor(sample['image'], dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32)
        }


class DataAugmentation:
    def __init__(self, p=0.5, each_transform_p=0.5):
        self.p = p
        _p = each_transform_p
        self.transforms = [
            RandomFlip(p=_p),
            RandomRotation(p=_p),
            RandomTranslation(p=_p),
            GaussianNoise(p=_p),
            SaltAndPepperNoise(p=_p),
        ]
        self.to_tensor = ToTensorWrapper()

    def __call__(self, sample):
        if random.random() <= self.p:
            for transform in self.transforms:
                sample = transform(sample)
        sample = self.to_tensor(sample)
        return sample
