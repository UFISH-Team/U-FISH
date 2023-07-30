import os
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.log import logger

if T.TYPE_CHECKING:
    from .unet.model import UNet
    from matplotlib.figure import Figure


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
        self._cuda = cuda
        self.model: T.Optional["UNet"] = None
        self.default_weight_file = default_weight_file
        self.store_base_url = BASE_STORE_URL
        self.local_store_path = Path(
            os.path.expanduser(local_store_path))

    def _init_model(self) -> None:
        import torch
        from .unet.model import UNet
        self.model = UNet()
        self.cuda = False
        if self._cuda:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.cuda = True
                logger.info('CUDA is available, using GPU.')
            else:
                logger.warning('CUDA is not available, using CPU.')
        else:
            logger.info('CUDA is not used, using CPU.')

    def load_weights(self, weights_path: T.Union[Path, str]) -> None:
        """Load weights from a local file.

        Args:
            weights_path: The path to the weights file."""
        import torch
        self._init_model()
        assert self.model is not None
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
        import torch
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
        if self.model is None:
            raise RuntimeError('Model is not initialized.')
        import torch
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
            return_enhanced_img: bool = False,
            ) -> T.Union[pd.DataFrame, T.Tuple[pd.DataFrame, np.ndarray]]:
        """Predict the spots in a 2D image.

        Args:
            img: The 2D image to predict.
            cc_size_thresh: Connected component size threshold.
            return_enhanced_img: Whether to return the enhanced image.

        Returns:
            spots_df: A pandas dataframe containing the spots.
            enhanced_img: The enhanced image. if return_enhanced_img is True.
        """
        assert img.ndim == 2, 'Image must be 2D.'
        img = img.astype(np.float32)
        from .calling import call_spots
        enhanced_img = self.enhance_img(img)
        df = call_spots(enhanced_img, cc_size_thresh)
        if return_enhanced_img:
            return df, enhanced_img
        else:
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
        from .utils.metrics import compute_metrics
        axis_names = list(pred.columns)
        axis_cols = [n for n in axis_names if n.startswith('axis')]
        pred = pred[axis_cols].values
        true = true[axis_cols].values
        metrics = compute_metrics(
            pred, true, mdist=mdist)
        return metrics

    def plot_result(
            self,
            img: np.ndarray,
            pred: pd.DataFrame,
            fig_size: T.Tuple[int, int] = (10, 10),
            image_cmap: str = 'gray',
            marker_size: int = 20,
            marker_color: str = 'red',
            marker_style: str = 'x',
            ) -> "Figure":
        """Plot the prediction result.

        Args:
            img: The image to plot.
            pred: The predicted spots.
            fig_size: The figure size.
            image_cmap: The colormap for the image.
            marker_size: The marker size.
            marker_color: The marker color.
            marker_style: The marker style.
        """
        from .utils.plot import Plot2d
        plt2d = Plot2d()
        plt2d.default_figsize = fig_size
        plt2d.default_marker_size = marker_size
        plt2d.default_marker_color = marker_color
        plt2d.default_marker_style = marker_style
        plt2d.default_imshow_cmap = image_cmap
        plt2d.new_fig()
        plt2d.image(img)
        plt2d.spots(pred.values)
        return plt2d.fig

    def plot_evaluate(
            self,
            img: np.ndarray,
            pred: pd.DataFrame,
            true: pd.DataFrame,
            cutoff: float = 3.0,
            fig_size: T.Tuple[int, int] = (10, 10),
            image_cmap: str = 'gray',
            marker_size: int = 20,
            tp_color: str = 'green',
            fp_color: str = 'red',
            fn_color: str = 'yellow',
            tp_marker: str = 'x',
            fp_marker: str = 'x',
            fn_marker: str = 'x',
            ) -> "Figure":
        """Plot the prediction result.

        Args:
            img: The image to plot.
            pred: The predicted spots.
            true: The true spots.
            cutoff: The maximum distance to consider a spot as a true positive.
            fig_size: The figure size.
            image_cmap: The colormap for the image.
            marker_size: The marker size.
            tp_color: The color for true positive.
            fp_color: The color for false positive.
            fn_color: The color for false negative.
            tp_marker_style: The marker style for true positive.
            fp_marker_style: The marker style for false positive.
            fn_marker_style: The marker style for false negative.
        """
        from .utils.plot import Plot2d
        plt2d = Plot2d()
        plt2d.default_figsize = fig_size
        plt2d.default_marker_size = marker_size
        plt2d.default_imshow_cmap = image_cmap
        plt2d.new_fig()
        plt2d.image(img)
        plt2d.evaluate_result(
            pred.values, true.values,
            cutoff=cutoff,
            tp_color=tp_color,
            fp_color=fp_color,
            fn_color=fn_color,
            tp_marker=tp_marker,
            fp_marker=fp_marker,
            fn_marker=fn_marker,
        )
        return plt2d.fig
