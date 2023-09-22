import sys
import typing as T
from os.path import splitext
from pathlib import Path

from .utils.log import logger


class UFishCLI():
    def __init__(
            self,
            cuda: bool = True,
            local_store_path: str = '~/.ufish/',
            weights_file_name: T.Optional[str] = None,
            ):
        from .api import UFish
        self._ufish = UFish(
            cuda=cuda,
            default_weights_file=weights_file_name,
            local_store_path=local_store_path,
        )
        self._weights_loaded = False

    def set_logger(
            self,
            log_file: T.Optional[str] = None,
            level: str = 'INFO'):
        """Set the log level."""
        logger.remove()
        logger.add(
            sys.stderr, level=level,
        )
        if log_file is not None:
            logger.info(f'Logging to {log_file}')
            logger.add(
                log_file, level=level,
            )
        return self

    def init_model(
            self,
            model_type: str = 'unet',
            **kwargs):
        """Initialize the model.

        Args:
            model_type: The type of the model. 'unet' or 'fcn'.
            kwargs: The keyword arguments for the model.
        """
        self._ufish.init_model(
            model_type=model_type,
            **kwargs
        )
        return self

    def convert_to_onnx(
            self,
            output_path: T.Union[Path, str],) -> None:
        """Convert the model to ONNX format.

        Args:
            output_path: The path to the output ONNX file.
        """
        self._ufish.convert_to_onnx(output_path)

    def load_weights(
            self,
            weights_path: T.Optional[str] = None,
            weights_file: T.Optional[str] = None,
            max_retry: int = 8,
            force_download: bool = False,
            ):
        """Load weights from a local file or the internet.

        Args:
            weights_path: The path to the weights file.
            weights_file: The name of the weights file on the internet.
                See https://huggingface.co/GangCaoLab/U-FISH/tree/main
                for available weights files.
            max_retry: The maximum number of retries to download the weights.
            force_download: Whether to force download the weights.
        """
        self._ufish.load_weights(
            weights_path=weights_path,
            weights_file=weights_file,
            max_retry=max_retry,
            force_download=force_download,
        )
        self._weights_loaded = True
        return self

    def call_spots(
            self,
            enhanced_img_path: str,
            output_csv_path: str,
            method: str = 'local_maxima',
            **kwargs,
            ):
        """Call spots by finding local maxima.

        Args:
            enhanced_img_path: Path to the enhanced image.
            output_csv_path: Path to the output csv file.
            method: The method to use for calling spots.
            kwargs: The keyword arguments for the method.
        """
        from skimage.io import imread
        img = imread(enhanced_img_path)
        logger.info(f'Calling spots in {enhanced_img_path}')
        logger.info(f'Method: {method}, Parameters: {kwargs}')
        pred_df = self._ufish.call_spots(
            img, method=method, **kwargs)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')

    def predict(
            self,
            input_img_path: str,
            output_csv_path: str,
            enhanced_output_path: T.Optional[str] = None,
            chunking: bool = False,
            chunk_size: T.Optional[T.Tuple[int, ...]] = None,
            axes: T.Optional[str] = None,
            blend_3d: bool = True,
            batch_size: int = 4,
            spot_calling_method: str = 'local_maxima',
            **kwargs,
            ):
        """Predict spots in image.

        Args:
            input_img_path: Path to the input image.
                Supported formats: tif, zarr, n5, ngff(ome-tiff).
            output_csv_path: Path to the output csv file.
            enhanced_output_path: Path to the enhanced image.
                If None, will not save the enhanced image.
                Supported formats: tif, zarr, n5, ngff(ome-tiff).
            chunking: Whether to use chunking for inference.
                If True, will infer the image chunk by chunk.
            chunk_size: The chunk size for processing.
                For example, (1, 512, 512) for a 3D image,
                (512, 512) for a 2D image.
                Using 'image' as a dimension will use the whole image
                as a chunk. For example, (1, 'image', 'image') for a 3D image,
                ('image', 'image', 'image', 512, 512) for a 5D image.
                If None, will use the default chunk size.
            axes: The axes of the image.
                For example, 'czxy' for a 4D image,
                'yx' for a 2D image.
                If None, will try to infer the axes from the shape.
            blend_3d: Whether to blend the 3D image.
                Used only when the image contains a z axis.
                If True, will blend the 3D enhanced images along
                the z, y, x axes.
            batch_size: The batch size for inference.
                Used only when the image dimension is 3 or higher.
            spots_calling_method: The method to use for spot calling.
            kwargs: Other arguments for the spot calling function.
        """
        from .utils.img import (
            open_for_read, open_enhimg_storage, save_enhimg,
            infer_img_axes,
        )
        if not self._weights_loaded:
            self.load_weights()
        logger.info(f'Predicting {input_img_path}')
        img = open_for_read(input_img_path)
        enhanced, tmp_ehn_path = open_enhimg_storage(
            enhanced_output_path, img.shape)
        if axes is None:
            logger.info("Axes not specified, infering from image shape.")
            axes = infer_img_axes(img.shape)
            logger.info(f"Infered axes: {axes}, image shape: {img.shape}")
        if chunking:
            pred_df, enhanced = self._ufish.predict_chunks(
                img, enh_img=enhanced,
                axes=axes, blend_3d=blend_3d,
                batch_size=batch_size,
                chunk_size=chunk_size,
                spots_calling_method=spot_calling_method,
                **kwargs)
        else:
            pred_df, enhanced = self._ufish.predict(
                img, enh_img=enhanced,
                axes=axes, blend_3d=blend_3d,
                batch_size=batch_size,
                spots_calling_method=spot_calling_method,
                **kwargs)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')
        if enhanced_output_path is not None:
            save_enhimg(
                enhanced, tmp_ehn_path, enhanced_output_path, axes
            )

    def predict_imgs(
            self,
            input_path: str,
            output_dir: str,
            data_base_dir: T.Optional[str] = None,
            img_glob: T.Optional[str] = None,
            save_enhanced_img: bool = True,
            table_suffix: str = '.pred.csv',
            enhanced_suffix: str = '.enhanced.tif',
            **kwargs,
            ):
        """Predict spots in a directory of 2d images.

        Args:
            input_path: Path to the input directory or
                the meta csv file.
            output_dir: Path to the output directory.
            data_base_dir: The base directory of the dataset.
                Only used when input_path is a meta csv file.
            img_glob: The glob pattern for the images.
            save_enhanced_img: Whether to save the enhanced image.
            table_suffix: The suffix for the output table.
            enhanced_suffix: The suffix for the enhanced image.
            kwargs: Other arguments for predict function.
        """
        if not self._weights_loaded:
            self.load_weights()
        if input_path.endswith('.csv'):
            import pandas as pd
            meta_df = pd.read_csv(input_path)
            base_dir = Path(data_base_dir)
            input_imgs = [base_dir / p for p in meta_df['image_path']]
        else:
            if img_glob is None:
                img_glob = '*'
            else:
                img_glob = img_glob.strip()
            in_dir_path = Path(input_path)
            input_imgs = list(in_dir_path.glob(img_glob))
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Predicting images in {input_path}')
        logger.info(f'Saving results to {out_dir_path}')
        for i, in_path in enumerate(input_imgs):
            logger.info(f'({i+1}/{len(input_imgs)}) Predicting {in_path}')
            input_prefix = splitext(in_path.name)[0]
            output_path = out_dir_path / (input_prefix + table_suffix)
            enhanced_img_path = out_dir_path / (input_prefix + enhanced_suffix)
            if enhanced_img_path.exists():
                logger.info(
                    f'Enhanced image {enhanced_img_path} exists, ' +
                    'skipping enhancement.')
                self.call_spots(
                    str(enhanced_img_path),
                    str(output_path),
                    **kwargs
                )
            else:
                self.predict(
                    str(in_path),
                    str(output_path),
                    str(enhanced_img_path) if save_enhanced_img else None,
                    **kwargs
                )

    def plot_2d_pred(
            self,
            image_path: str,
            pred_csv_path: str,
            fig_save_path: T.Optional[str] = None,
            **kwargs
            ):
        """Plot the predicted spots on the image."""
        import pandas as pd
        from skimage.io import imread
        import matplotlib.pyplot as plt
        img = imread(image_path)
        pred_df = pd.read_csv(pred_csv_path)
        fig = self._ufish.plot_result(
            img, pred_df, **kwargs
        )
        if fig_save_path is not None:
            fig.savefig(fig_save_path)
            logger.info(f'Saved figure to {fig_save_path}.')
        else:
            plt.show()

    def plot_2d_eval(
            self,
            image_path: str,
            pred_csv_path: str,
            true_csv_path: str,
            cutoff: float = 3.0,
            fig_save_path: T.Optional[str] = None,
            **kwargs
            ):
        """Plot the evaluation result."""
        import pandas as pd
        from skimage.io import imread
        import matplotlib.pyplot as plt
        img = imread(image_path)
        pred_df = pd.read_csv(pred_csv_path)
        gt_df = pd.read_csv(true_csv_path)
        fig = self._ufish.plot_evaluate(
            img, pred_df, gt_df, cutoff=cutoff, **kwargs)
        if fig_save_path is not None:
            fig.savefig(fig_save_path)
            logger.info(f'Saved figure to {fig_save_path}.')
        else:
            plt.show()

    def plot_2d_evals(
            self,
            pred_dir: str,
            true_dir: str,
            image_dir: str,
            output_dir: str,
            pred_glob: str = '*.pred.csv',
            true_glob: str = '*.csv',
            image_glob: str = '*.tif',
            cutoff: float = 3.0,
            output_suffix: str = '.png',
            **kwargs
            ):
        """Plot the evaluation result for a directory of images."""
        pred_dir_path = Path(pred_dir)
        true_dir_path = Path(true_dir)
        image_dir_path = Path(image_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        pred_csvs = list(pred_dir_path.glob(pred_glob))
        true_csvs = list(true_dir_path.glob(true_glob))
        image_paths = list(image_dir_path.glob(image_glob))
        common_names = set([p.name.split('.')[0] for p in pred_csvs]) & \
            set([p.name.split('.')[0] for p in true_csvs]) & \
            set([p.name.split('.')[0] for p in image_paths])
        logger.info(f'Plotting {len(common_names)} images')
        for idx, name in enumerate(common_names):
            logger.info(f'Plotting ({idx+1}/{len(common_names)}) {name}')
            pred_csv_path = pred_dir_path / (name + '.pred.csv')
            true_csv_path = true_dir_path / (name + '.csv')
            image_path = image_dir_path / (name + '.tif')
            output_path = output_dir_path / (name + output_suffix)
            try:
                self.plot_2d_eval(
                    str(image_path),
                    str(pred_csv_path),
                    str(true_csv_path),
                    cutoff=cutoff,
                    fig_save_path=str(output_path),
                    **kwargs
                )
            except Exception as e:
                logger.error(f'Error plotting {name}: {e}')

    def evaluate_imgs(
            self,
            pred_dir: str,
            true_dir: str,
            output_table_path: str,
            deepblink_metric: bool = False,
            pred_glob: str = '*.pred.csv',
            true_glob: str = '*.csv',
            ):
        """Evaluate the predicted spots.

        Args:
            pred_dir: Path to the directory containing the predicted spots.
            true_dir: Path to the directory containing the true spots.
            output_table_path: Path to the output table.
            deepblink_metric: Whether to use the DeepBlink metric.
            pred_glob: The glob pattern for the predicted spots.
            true_glob: The glob pattern for the true spots.
        """
        import pandas as pd
        pred_dir_path = Path(pred_dir)
        true_dir_path = Path(true_dir)
        pred_csvs = list(pred_dir_path.glob(pred_glob))
        true_csvs = list(true_dir_path.glob(true_glob))
        common_names = set([p.name.split('.')[0] for p in pred_csvs]) & \
            set([p.name.split('.')[0] for p in true_csvs])
        logger.info(f'Evaluating {len(common_names)} images')
        out = {
            'true_csv': [],
            'pred_csv': [],
            'source': [],
            'f1(cutoff=3)': [],
            'pred num': [],
            'true num': [],
            'mean distance': [],
        }
        if deepblink_metric:
            out['f1 integral'] = []
        else:
            out['true positive'] = []
            out['false negative'] = []
            out['false positive'] = []
            out['recall'] = []
            out['precision'] = []
        for idx, name in enumerate(common_names):
            pred_csv_path = pred_dir_path / (name + '.pred.csv')
            true_csv_path = true_dir_path / (name + '.csv')
            pred_df = pd.read_csv(pred_csv_path)
            true_df = pd.read_csv(true_csv_path)
            source = name.split("_")[0]
            out['true_csv'].append(true_csv_path)
            out['pred_csv'].append(pred_csv_path)
            out['source'].append(source)
            out['pred num'].append(len(pred_df))
            out['true num'].append(len(true_df))
            if not deepblink_metric:
                res = self._ufish.evaluate_result(
                    pred_df, true_df, cutoff=3.0)
                out['f1(cutoff=3)'].append(res['f1'])
                out['true positive'].append(res['true_positive'])
                out['false negative'].append(res['false_negative'])
                out['false positive'].append(res['false_positive'])
                out['recall'].append(res['recall'])
                out['precision'].append(res['precision'])
                out['mean distance'].append(res['mean_dist'])
                logger.info(
                    f'Evaluated ({idx+1}/{len(common_names)}) {name}, ' +
                    f'f1(cutoff=3): {out["f1(cutoff=3)"][-1]:.4f}, ' +
                    f'pred num: {out["pred num"][-1]}, ' +
                    f'true num: {out["true num"][-1]}, ' +
                    f'true positive: {out["true positive"][-1]}, ' +
                    f'false negative: {out["false negative"][-1]}, ' +
                    f'false positive: {out["false positive"][-1]}, ' +
                    f'recall: {out["recall"][-1]:.4f}, ' +
                    f'precision: {out["precision"][-1]:.4f}, ' +
                    f'mean distance: {out["mean distance"][-1]:.4f}'
                )
            else:
                res = self._ufish.evaluate_result_dp(
                    pred_df, true_df, mdist=3.0)
                out['f1(cutoff=3)'].append(res['f1_score'].iloc[-1])
                out['f1 integral'].append(res['f1_integral'].iloc[0])
                out['mean distance'].append(res['mean_euclidean'].iloc[0])
                logger.info(
                    f'Evaluated ({idx+1}/{len(common_names)}) {name}, ' +
                    f'f1(cutoff=3): {out["f1(cutoff=3)"][-1]:.4f}, ' +
                    f'f1 integral: {out["f1 integral"][-1]:.4f}, ' +
                    f'mean distance: {out["mean distance"][-1]:.4f}, ' +
                    f'pred num: {out["pred num"][-1]}, ' +
                    f'true num: {out["true num"][-1]}'
                )
        out_df = pd.DataFrame(out)
        mean_f1 = out_df['f1(cutoff=3)'].mean()
        logger.info(f'Mean f1(cutoff=3): {mean_f1:.4f}')
        logger.info(f'Saving results to {output_table_path}')
        out_df.to_csv(output_table_path, index=False)

    def train(
            self,
            train_path: str,
            valid_path: str,
            root_dir: T.Optional[str] = None,
            img_glob: str = '*.tif',
            coord_glob: str = '*.csv',
            target_process: T.Optional[str] = 'gaussian',
            loss_type: str = 'DiceRMSELoss',
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
            coord_glob: The glob pattern for the coordinate files.
            target_process: The target image processing method.
                'gaussian' or 'dialation' or None.
                If None, no processing will be applied.
                default 'gaussian'.
            loss_type: The loss function type.
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
        self._ufish.train(
            train_path=train_path,
            valid_path=valid_path,
            root_dir=root_dir,
            img_glob=img_glob,
            coord_glob=coord_glob,
            target_process=target_process,
            loss_type=loss_type,
            loader_workers=loader_workers,
            data_argu=data_argu,
            argu_prob=argu_prob,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            summary_dir=summary_dir,
            model_save_dir=model_save_dir,
            save_period=save_period,
        )
