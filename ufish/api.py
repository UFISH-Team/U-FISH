import os
import time
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.log import logger

if T.TYPE_CHECKING:
    from torch import nn
    from matplotlib.figure import Figure
    import onnxruntime


BASE_STORE_URL = 'https://huggingface.co/GangCaoLab/U-FISH/resolve/main/'


class UFish():
    def __init__(
            self, cuda: bool = True,
            default_weights_file: str = 'v2.0-alldata-unet_d2_b32.onnx',
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
        self.model: T.Optional["nn.Module"] = None
        self.ort_session: T.Optional["onnxruntime.InferenceSession"] = None
        self.default_weights_file = default_weights_file
        self.store_base_url = BASE_STORE_URL
        self.local_store_path = Path(
            os.path.expanduser(local_store_path))

    def init_model(
            self,
            model_type: str = 'unet',
            **kwargs) -> None:
        """Initialize the model.

        Args:
            model_type: The type of the model. 'unet' or 'fcn'.
            **kwargs: The keyword arguments for the model.
        """
        import torch
        if model_type == 'unet':
            from .model.network import UNet
            self.model = UNet(**kwargs)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f'Initializing {model_type} model with setting: \n' +
            f'{kwargs}'
        )
        logger.info(f'Number of parameters: {params}')
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

    def convert_to_onnx(
            self,
            output_path: T.Union[Path, str],) -> None:
        """Convert the model to ONNX format.

        Args:
            output_path: The path to the output ONNX file.
        """
        if self.model is None:
            raise RuntimeError('Model is not initialized.')
        self._turn_on_infer_mode(trace_model=True)
        import torch
        import torch.onnx
        output_path = str(output_path)
        logger.info(
            f'Converting model to ONNX format, saving to {output_path}')
        device = torch.device('cuda' if self.cuda else 'cpu')
        inp = torch.rand(1, 1, 512, 512).to(device)
        dyn_axes = {0: 'batch_size', 2: 'y', 3: 'x'}
        torch.onnx.export(
            self.model, inp, output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            do_constant_folding=True,
            dynamic_axes={
                'input': dyn_axes,
                'output': dyn_axes,
            },
        )

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

    def load_weights(
            self,
            path: T.Union[Path, str],
            ) -> None:
        """Load weights from a local file.
        The file can be a .pth file or an .onnx file.

        Args:
            path: The path to the weights file.
        """
        path = str(path)
        if path.endswith('.pth'):
            self._load_pth_file(path)
        elif path.endswith('.onnx'):
            self._load_onnx(path)
        else:
            raise ValueError(
                'Weights file must be a pth file or an onnx file.')

    def _load_pth_file(self, path: T.Union[Path, str]) -> None:
        """Load weights from a local file.

        Args:
            path: The path to the pth weights file."""
        import torch
        if self.model is None:
            self.init_model()
        assert self.model is not None
        path = str(path)
        logger.info(f'Loading weights from {path}')
        device = torch.device('cuda' if self.cuda else 'cpu')
        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)

    def _load_onnx(
            self,
            onnx_path: T.Union[Path, str],
            providers: T.Optional[T.List[str]] = None,
            ) -> None:
        """Load weights from a local ONNX file,
        and create an onnxruntime session.

        Args:
            onnx_path: The path to the ONNX file.
            providers: The providers to use.
        """
        import onnxruntime
        onnx_path = str(onnx_path)
        logger.info(f'Loading ONNX from {onnx_path}')
        if self._cuda:
            providers = providers or ['CUDAExecutionProvider']
        else:
            providers = providers or ['CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(
            onnx_path, providers=providers)

    def infer(self, img: np.ndarray) -> np.ndarray:
        """Infer the image using the U-Net model."""
        if self.ort_session is not None:
            output = self._infer_onnx(img)
        elif self.model is not None:
            output = self._infer_torch(img)
        else:
            raise RuntimeError(
                'Both torch model and ONNX model are not initialized.')
        return output

    def _infer_torch(self, img: np.ndarray) -> np.ndarray:
        """Infer the image using the torch model."""
        self._turn_on_infer_mode()
        import torch
        tensor = torch.from_numpy(img).float()
        if self.cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            output = self.model(tensor)
        output = output.detach().cpu().numpy()
        return output

    def _infer_onnx(self, img: np.ndarray) -> np.ndarray:
        """Infer the image using the ONNX model."""
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        return output

    def enhance_img(self, img: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """Enhance the image using the U-Net model."""
        from .utils.misc import scale_image
        img = scale_image(img, warning=True)
        if img.ndim == 2:
            output = self._enhance_img2d(img)
        elif img.ndim == 3:
            output = self._enhance_img3d(img, batch_size=batch_size)
        else:
            raise ValueError('Image must be 2D or 3D.')
        return output

    def _enhance_img2d(self, img: np.ndarray) -> np.ndarray:
        """Enhance a 2D image."""
        output = self.infer(img[np.newaxis, np.newaxis])[0, 0]
        return output

    def _enhance_img3d(
            self, img: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """Enhance a 3D image."""
        output = np.zeros_like(img)
        for i in range(0, output.shape[0], batch_size):
            _slice = img[i:i+batch_size][:, np.newaxis]
            output[i:i+batch_size] = self.infer(_slice)[:, 0]
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
            intensity_threshold: float = 0.5,
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

    def _pred_2d_or_3d(
            self, img: np.ndarray,
            intensity_threshold: float = 0.5,
            batch_size: int = 4,
            ) -> T.Tuple[pd.DataFrame, np.ndarray]:
        """Predict the spots in a 2D or 3D image. """
        assert img.ndim in (2, 3), 'Image must be 2D or 3D.'
        enhanced_img = self.enhance_img(img, batch_size=batch_size)
        df = self.call_spots_local_maxima(
            enhanced_img, connectivity=2,
            intensity_threshold=intensity_threshold,
        )
        return df, enhanced_img

    def predict(
            self, img: np.ndarray,
            axes: T.Optional[str] = None,
            intensity_threshold: float = 0.5,
            batch_size: int = 4,
            ) -> T.Tuple[pd.DataFrame, np.ndarray]:
        """Predict the spots in an image.

        Args:
            img: The image to predict, it should be a multi dimensional array.
                For example, shape (c, z, y, x) for a 4D image,
                shape (z, y, x) for a 3D image,
                shape (y, x) for a 2D image.
            axes: The axes of the image.
                For example, 'czxy' for a 4D image,
                'yx' for a 2D image.
                If None, will try to infer the axes from the shape.
            intensity_threshold: The threshold for the intensity.
            batch_size: The batch size for inference.
                Used only when the image dimension is 3 or higher.
        """
        from .utils.misc import (
            infer_img_axes, check_img_axes,
            map_predfunc_to_img
        )
        from functools import partial
        if axes is None:
            logger.info("Axes not specified, infering from image shape.")
            axes = infer_img_axes(img.shape)
            logger.info(f"Infered axes: {axes}, image shape: {img.shape}")
        check_img_axes(axes)
        predfunc = partial(
            self._pred_2d_or_3d,
            intensity_threshold=intensity_threshold,
            batch_size=batch_size)
        df, enhanced_img = map_predfunc_to_img(
            predfunc, img, axes)
        return df, enhanced_img

    def evaluate_result_dp(
            self,
            pred: pd.DataFrame,
            true: pd.DataFrame,
            mdist: float = 3.0,
            ) -> pd.DataFrame:
        """Evaluate the prediction result using deepblink metrics.

        Args:
            pred: The predicted spots.
            true: The true spots.
            mdist: The maximum distance to consider a spot as a true positive.

        Returns:
            A pandas dataframe containing the evaluation metrics."""
        from .utils.metrics_deepblink import compute_metrics
        axis_names = list(pred.columns)
        axis_cols = [n for n in axis_names if n.startswith('axis')]
        pred = pred[axis_cols].values
        true = true[axis_cols].values
        metrics = compute_metrics(
            pred, true, mdist=mdist)
        return metrics

    def evaluate_result(
            self,
            pred: pd.DataFrame,
            true: pd.DataFrame,
            cutoff: float = 3.0,
            ) -> float:
        """Calculate the F1 score of the prediction result.

        Args:
            pred: The predicted spots.
            true: The true spots.
            cutoff: The maximum distance to consider a spot as a true positive.
        """
        from .utils.metrics import compute_metrics
        res = compute_metrics(pred.values, true.values, cutoff=cutoff)
        return res

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

    def _load_dataset(
            self,
            path: str,
            root_dir_path: T.Optional[str] = None,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            process_func=None,
            transform=None,
            ):
        """Load a dataset from a path."""
        from .data import FISHSpotsDataset
        _path = Path(path)
        if _path.is_dir():
            if root_dir_path is not None:
                logger.info(f"Dataset's root directory: {root_dir_path}")
                _path = Path(root_dir_path) / _path
            logger.info(f"Loading dataset from dir: {_path}")
            logger.info(
                f'Image glob: {img_glob}, Coordinate glob: {coord_glob}')
            _path_str = str(_path)
            dataset = FISHSpotsDataset.from_dir(
                _path_str, _path_str,
                img_glob=img_glob, coord_glob=coord_glob,
                process_func=process_func, transform=transform)
        else:
            logger.info(f"Loading dataset using meta csv: {_path}")
            assert _path.suffix == '.csv', \
                "Meta file must be a csv file."
            root_dir = root_dir_path or _path.parent
            logger.info(f'Data root directory: {root_dir}')
            dataset = FISHSpotsDataset.from_meta_csv(
                root_dir=root_dir, meta_csv_path=str(_path),
                process_func=process_func, transform=transform)
        return dataset

    def train(
            self,
            train_path: str,
            valid_path: str,
            root_dir: T.Optional[str] = None,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            target_process: T.Optional[str] = 'gaussian',
            loader_workers: int = 4,
            data_argu: bool = False,
            argu_prob: float = 0.5,
            num_epochs: int = 50,
            batch_size: int = 8,
            lr: float = 1e-3,
            summary_dir: str = "runs/unet",
            model_save_dir: str = "./models",
            save_period: int = 5,
            ):
        """Train the U-Net model.

        Args:
            train_path: The path to the training dataset.
                Path to a directory containing images and coordinates,
                or a meta csv file.
            valid_path: The path to the validation dataset.
                Path to a directory containing images and coordinates,
                or a meta csv file.
            root_dir: The root directory of the dataset.
                If using meta csv, the root directory of the dataset.
            img_glob: The glob pattern for the image files.
            coord_glob: The glob pattern for the coordinate files.
            target_process: The target image processing method.
                'gaussian' or 'dialation' or None.
                If None, no processing will be applied.
                default 'gaussian'.
            loader_workers: The number of workers to use for the data loader.
            data_argu: Whether to use data augmentation.
            argu_prob: The probability to use data augmentation.
            num_epochs: The number of epochs to train.
            batch_size: The batch size.
            lr: The learning rate.
            summary_dir: The directory to save the TensorBoard summary to.
            model_save_dir: The directory to save the model to.
            save_period: Save the model every `save_period` epochs.
        """
        from .model.train import train_on_dataset
        from .data import FISHSpotsDataset
        if self.model is None:
            logger.info('Model is not initialized. Will initialize a new one.')
            self.init_model()
        assert self.model is not None

        if data_argu:
            logger.info(
                'Using data augmentation. ' +
                f'Probability: {argu_prob}'
            )
            from .data import DataAugmentation
            transform = DataAugmentation(p=argu_prob)
        else:
            transform = None

        logger.info(f'Using {target_process} as target process.')
        if target_process == 'gaussian':
            process_func = FISHSpotsDataset.gaussian_filter
        elif target_process == 'dialation':
            process_func = FISHSpotsDataset.dialate_mask
        else:
            process_func = None

        logger.info(f"Loading training dataset from {train_path}")
        train_dataset = self._load_dataset(
            train_path, root_dir_path=root_dir,
            img_glob=img_glob, coord_glob=coord_glob,
            process_func=process_func, transform=transform,
        )
        logger.info(f"Loading validation dataset from {valid_path}")
        valid_dataset = self._load_dataset(
            valid_path, root_dir_path=root_dir,
            img_glob=img_glob, coord_glob=coord_glob,
            process_func=process_func,
        )
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
            loader_workers=loader_workers,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            summary_dir=summary_dir,
            model_save_dir=model_save_dir,
            save_period=save_period,
        )
