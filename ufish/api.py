import os
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from .unet.model import UNet
from .calling import call_spots
from .utils.metrics import compute_metrics


BASE_STORE_URL = 'https://huggingface.co/GangCaoLab/U-FISH/resolve/main/'


class UFish():
    def __init__(
            self, cuda: bool = True,
            default_weight_file: str = 'v1-for_benchmark.pth',
            local_store_path: str = '~/.ufish/'
            ) -> None:
        """
        Args:
            cuda: Whether to use GPU.
            default_weight_file: The default weight file to use.
            local_store_path: The local path to store the weights.
        """
        self.model = UNet()
        self.cuda = False
        if cuda:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.cuda = True
                logger.info('CUDA is available, using GPU.')
            else:
                logger.warning('CUDA is not available, using CPU.')
        else:
            logger.info('CUDA is not used, using CPU.')
        self.store_base_url = BASE_STORE_URL
        self.default_weight_file = default_weight_file
        self.local_store_path = Path(
            os.path.expanduser(local_store_path))

    def load_weights(self, weights_path: T.Union[Path, str]) -> None:
        """Load weights from a local file.

        Args:
            weights_path: The path to the weights file."""
        weights_path = str(weights_path)
        logger.info(f'Loading weights from {weights_path}.')
        device = torch.device('cuda' if self.cuda else 'cpu')
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)

    def load_weights_from_internet(
            self, weight_file: T.Optional[str] = None) -> None:
        """Load weights from the huggingface repo.

        Args:
            weight_file: The weight file name to load.
        """
        weight_file = weight_file or self.default_weight_file
        weight_url = self.store_base_url + weight_file
        local_weight_path = self.local_store_path / weight_file
        if local_weight_path.exists():
            logger.info(
                f'Local weights {local_weight_path} exists, '
                'skip downloading.'
            )
        else:
            logger.info(
                f'Downloading weights from {weight_url}, '
                f'storing to {local_weight_path}.')
            local_weight_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weight_url, local_weight_path)
        self.load_weights(local_weight_path)

    def enhance_img(self, img: np.ndarray) -> np.ndarray:
        """Enhance the image using the U-Net model."""
        assert img.ndim == 2, 'Image must be 2D.'
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        if self.cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            output = self.model(tensor)
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        return output

    def pred_2d(
            self, img: np.ndarray,
            cc_size_thresh: int = 18,
            ) -> pd.DataFrame:
        """Predict the spots in a 2D image.

        Args:
            img: The 2D image to predict.
            cc_size_thresh: Connected component size threshold.

        Returns:
            A pandas dataframe with columns ['axis-0', 'axis-1']."""
        assert img.ndim == 2, 'Image must be 2D.'
        enhanced_img = self.enhance_img(img)
        df = call_spots(enhanced_img, cc_size_thresh)
        return df

    def evaluate_result(
            self,
            pred: pd.DataFrame,
            true: pd.DataFrame,
            mdist: float = 3.0,
            ) -> pd.DataFrame:
        """Evaluate the prediction result.

        Args:
            pred: The predicted spots.
            true: The true spots.
            mdist: The maximum distance to consider a spot as a true positive.

        Returns:
            A pandas dataframe containing the evaluation metrics."""
        axis_names = list(pred.columns)
        axis_cols = [n for n in axis_names if n.startswith('axis')]
        pred = pred[axis_cols].values
        true = true[axis_cols].values
        metrics = compute_metrics(
            pred, true, mdist=mdist)
        return metrics
