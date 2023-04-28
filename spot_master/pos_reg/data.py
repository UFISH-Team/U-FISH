import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class FISHSpotsDataset(Dataset):
    def __init__(
            self, meta_csv, root_dir, max_spots=8000):
        self.meta_data = pd.read_csv(meta_csv)
        self.root_dir = root_dir
        self.max_spots = max_spots

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 0])
        image = io.imread(img_path)
        image = np.expand_dims(image, axis=0)

        coord_path = os.path.join(self.root_dir, self.meta_data.iloc[idx, 1])
        coordinates = pd.read_csv(coord_path).values

        padded_coordinates = np.zeros((self.max_spots, 2), dtype=np.float32)
        n_spots = coordinates.shape[0]
        padded_coordinates[:n_spots, :] = coordinates[:self.max_spots, :]

        sample = {
            'image': image,
            'coordinates': padded_coordinates,
            'n_spots': n_spots
        }

        return sample
