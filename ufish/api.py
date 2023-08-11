import os
import time
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
            default_weights_file: str = 'v1.1-gaussian_target.pth',
            local_store_path: str = '~/.ufish/'
            ) -> None:
        """
        Args:
            cuda: Whether to use GPU.
            default_weight_file: The default weight file to use.
            local_store_path: The local path to store the weights.
        """
        self._cuda = cuda
        self._infer_mode = False
        self.model: T.Optional["UNet"] = None
        self.default_weights_file = default_weights_file
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

    def _turn_on_infer_mode(self, trace_model: bool = False) -> None:
        """Turn on the infer mode."""
        if self._infer_mode:
            return
        self._infer_mode = True
        self.model.eval()
        if trace_model:
            import torch
            device = next(self.model.parameters()).device
            inp = torch.rand(1, 1, 512, 512).to(device)
            self.model = torch.jit.trace(self.model, inp)

    def load_weights(self, weights_path: T.Union[Path, str]) -> None:
        """Load weights from a local file.

        Args:
            weights_path: The path to the weights file."""
        import torch
        self._init_model()
        assert self.model is not None
        weights_path = str(weights_path)
        logger.info(f'Loading weights from {weights_path}')
        device = torch.device('cuda' if self.cuda else 'cpu')
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)

    def load_weights_from_internet(
            self,
            weights_file: T.Optional[str] = None,
            max_retry: int = 8,
            force_download: bool = False,
            ) -> None:
        """Load weights from the huggingface repo.

        Args:
            weights_file: The name of the weights file on the internet.
                See https://huggingface.co/GangCaoLab/U-FISH/tree/main
                for available weights files.
            max_retry: The maximum number of retries.
            force_download: Whether to force download the weights.
        """
        import torch
        weights_file = weights_file or self.default_weights_file
        weight_url = self.store_base_url + weights_file
        local_weight_path = self.local_store_path / weights_file
        if local_weight_path.exists() and (not force_download):
            logger.info(
                f'Local weights {local_weight_path} exists, '
                'skip downloading.'
            )
        else:
            logger.info(
                f'Downloading weights from {weight_url}, '
                f'storing to {local_weight_path}')
            local_weight_path.parent.mkdir(parents=True, exist_ok=True)
            try_count = 0
            while try_count < max_retry:
                try:
                    torch.hub.download_url_to_file(
                        weight_url, local_weight_path)
                    break
                except Exception as e:
                    logger.warning(f'Error downloading weights: {e}')
                    try_count += 1
                    time.sleep(0.5)
            else:
                raise RuntimeError(
                    f'Error downloading weights from {weight_url}.')
        self.load_weights(local_weight_path)

    def enhance_img(self, img: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """Enhance the image using the U-Net model."""
        if self.model is None:
            raise RuntimeError('Model is not initialized.')
        self._turn_on_infer_mode()
        from .utils.misc import scale_image
        img = scale_image(img)
        if img.ndim == 2:
            output = self._enhance_img2d(img)
        elif img.ndim == 3:
            output = self._enhance_img3d(img, batch_size=batch_size)
        else:
            raise ValueError('Image must be 2D or 3D.')
        return output

    def _enhance_img2d(self, img: np.ndarray) -> np.ndarray:
        """Enhance a 2D image."""
        import torch
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        if self.cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            output = self.model(tensor)
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        return output

    def _enhance_img3d(
            self, img: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """Enhance a 3D image."""
        import torch
        tensor = torch.from_numpy(img).float().unsqueeze(1)
        if self.cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            for i in range(0, tensor.shape[0], batch_size):
                tensor[i:i+batch_size] = self.model(tensor[i:i+batch_size])
        output = tensor.squeeze(1).cpu().numpy()
        return output

    def call_spots_cc_center(
            self, enhanced_img: np.ndarray,
            binary_threshold: T.Union[str, float] = 'otsu',
            cc_size_thresh: int = 20,
            ) -> pd.DataFrame:
        """Call spots by finding the connected components' centroids.

        Args:
            enhanced_img: The enhanced image.
            binary_threshold: The threshold for binarizing the image.
            cc_size_thresh: Connected component size threshold.

        Returns:
            A pandas dataframe containing the spots.
        """
        assert enhanced_img.ndim in (2, 3), 'Image must be 2D or 3D.'
        from .calling import call_spots_cc_center
        df = call_spots_cc_center(
            enhanced_img,
            cc_size_threshold=cc_size_thresh,
            binary_threshold=binary_threshold,
        )
        return df

    def call_spots_local_maxima(
            self, enhanced_img: np.ndarray,
            connectivity: int = 2,
            intensity_threshold: float = 0.1,
            ) -> pd.DataFrame:
        """Call spots by finding the local maxima.

        Args:
            enhanced_img: The enhanced image.
            connectivity: The connectivity for the local maxima.
            intensity_threshold: The threshold for the intensity.

        Returns:
            A pandas dataframe containing the spots.
        """
        assert enhanced_img.ndim in (2, 3), 'Image must be 2D or 3D.'
        from skimage.morphology import local_maxima
        mask = local_maxima(enhanced_img, connectivity=connectivity)
        mask = mask & (enhanced_img > intensity_threshold)
        peaks = np.array(np.where(mask)).T
        df = pd.DataFrame(
            peaks, columns=[f'axis-{i}' for i in range(mask.ndim)])
        return df

    def pred_2d(
            self, img: np.ndarray,
            connectivity: int = 2,
            intensity_threshold: float = 0.1,
            return_enhanced_img: bool = False,
            ) -> T.Union[pd.DataFrame, T.Tuple[pd.DataFrame, np.ndarray]]:
        """Predict the spots in a 2D image.

        Args:
            img: The 2D image to predict.
            connectivity: The connectivity for the local maxima.
            intensity_threshold: The threshold for the intensity.
            return_enhanced_img: Whether to return the enhanced image.

        Returns:
            spots_df: A pandas dataframe containing the spots.
            enhanced_img: The enhanced image. if return_enhanced_img is True.
        """
        assert img.ndim == 2, 'Image must be 2D.'
        enhanced_img = self.enhance_img(img)
        df = self.call_spots_local_maxima(
            enhanced_img,
            connectivity=connectivity,
            intensity_threshold=intensity_threshold,
        )
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

    def train(
            self,
            train_dir: str,
            valid_dir: str,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            target_process: str = 'gaussian',
            data_argu: bool = False,
            num_epochs: int = 50,
            batch_size: int = 8,
            lr: float = 1e-4,
            summary_dir: str = "runs/unet",
            model_save_path: str = "best_unet_model.pth",
            only_save_best: bool = True,
            ):
        """Train the U-Net model.

        Args:
            train_dir: The directory containing the training images
                and coordinates.
            valid_dir: The directory containing the validation images
                and coordinates.
            img_glob: The glob pattern for the image files.
            coord_glob: The glob pattern for the coordinate files.
            target_process: The target image processing method.
                'gaussian' or 'dialation'. default 'gaussian'.
            data_argu: Whether to use data augmentation.
            num_epochs: The number of epochs to train.
            batch_size: The batch size.
            lr: The learning rate.
            summary_dir: The directory to save the TensorBoard summary to.
            model_save_path: The path to save the best model to.
            only_save_best: Whether to only save the best model.
        """
        from .unet.train import train_on_dataset
        from .unet.data import FISHSpotsDataset
        if self.model is None:
            logger.info('Model is not initialized. Will initialize a new one.')
            self._init_model()
        assert self.model is not None

        if data_argu:
            logger.info('Using data augmentation.')
            from .unet.data import composed_transform
            transform = composed_transform
        else:
            transform = None

        logger.info(f'Using {target_process} as target process.')
        if target_process == 'gaussian':
            process_func = FISHSpotsDataset.gaussian_filter
        else:
            process_func = FISHSpotsDataset.dialate_mask

        logger.info(f'Image glob: {img_glob}, Coordinate glob: {coord_glob}')
        logger.info(f'Loading training data from {train_dir}')
        train_dataset = FISHSpotsDataset.from_dir(
            train_dir, train_dir,
            img_glob=img_glob, coord_glob=coord_glob,
            process_func=process_func, transform=transform)
        logger.info(f'Loading validation data from {valid_dir}')
        valid_dataset = FISHSpotsDataset.from_dir(
            valid_dir, valid_dir,
            img_glob=img_glob, coord_glob=coord_glob,
            process_func=process_func, transform=transform)
        logger.info(
            f"Training dataset size: {len(train_dataset)}, "
            f"Validation dataset size: {len(valid_dataset)}"
        )
        logger.info(
            f"Number of epochs: {num_epochs}, "
            f"Batch size: {batch_size}, "
            f"Learning rate: {lr}"
        )
        train_on_dataset(
            self.model,
            train_dataset, valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            summary_dir=summary_dir,
            model_save_path=model_save_path,
            only_save_best=only_save_best,
        )
