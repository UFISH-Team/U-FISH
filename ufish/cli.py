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
            weights_file_name: str = 'v1.1-gaussian_target.pth',
            ):
        from .api import UFish
        self._ufish = UFish(
            cuda=cuda,
            default_weights_file=weights_file_name,
            local_store_path=local_store_path,
        )
        self._weights_loaded = False

    def set_log_level(
            self,
            level: str = 'INFO'):
        """Set the log level."""
        logger.remove()
        logger.add(
            sys.stderr, level=level,
        )
        return self

    def init_model(
            self,
            model_type: str = 'unet',
            depth: int = 3,
            base_channels: int = 64,
            ):
        """Initialize the model.

        Args:
            model_type: The type of the model. 'unet' or 'fcn'.
            depth: The depth of the network.
            base_channels: The number of base channels.
        """
        self._ufish.init_model(
            model_type=model_type,
            depth=depth,
            base_channels=base_channels,
        )
        return self

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
        if weights_path is not None:
            self._ufish.load_weights(weights_path)
        else:
            self._ufish.load_weights_from_internet(
                weights_file=weights_file,
                max_retry=max_retry,
                force_download=force_download,
            )
        self._weights_loaded = True
        return self

    def enhance_img(
            self,
            input_img_path: str,
            output_img_path: str,
            ):
        """Enhance an image."""
        from skimage.io import imread, imsave
        if not self._weights_loaded:
            self.load_weights()
        logger.info(f'Enhancing {input_img_path}')
        img = imread(input_img_path)
        enhanced = self._ufish.enhance_img(img)
        imsave(output_img_path, enhanced)
        logger.info(f'Saved enhanced image to {output_img_path}')

    def call_spots_cc_center(
            self,
            enhanced_img_path: str,
            output_csv_path: str,
            binary_threshold: T.Union[str, float] = 'otsu',
            cc_size_thresh: int = 20
            ):
        """Call spots by finding connected components
        and taking the centroids.

        Args:
            enhanced_img_path: Path to the enhanced image.
            output_csv_path: Path to the output csv file.
            binary_threshold: The threshold for binarizing the image.
            cc_size_thresh: Connected component size threshold.
        """
        from skimage.io import imread
        img = imread(enhanced_img_path)
        logger.info(f'Calling spots in {enhanced_img_path}')
        logger.info(
            f'Parameters: binary_threshold={binary_threshold}, ' +
            f'cc_size_thresh={cc_size_thresh}')
        pred_df = self._ufish.call_spots_cc_center(
            img,
            binary_threshold=binary_threshold,
            cc_size_thresh=cc_size_thresh)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')

    def call_spots_local_maxima(
            self,
            enhanced_img_path: str,
            output_csv_path: str,
            connectivity: int = 2,
            intensity_threshold: float = 0.1,
            ):
        """Call spots by finding local maxima.

        Args:
            enhanced_img_path: Path to the enhanced image.
            output_csv_path: Path to the output csv file.
            connectivity: The connectivity for finding local maxima.
            intensity_threshold: The threshold for the intensity.
        """
        from skimage.io import imread
        img = imread(enhanced_img_path)
        logger.info(f'Calling spots in {enhanced_img_path}')
        logger.info(
            f'Parameters: connectivity={connectivity}, ' +
            f'intensity_threshold={intensity_threshold}')
        pred_df = self._ufish.call_spots_local_maxima(
            img,
            connectivity=connectivity,
            intensity_threshold=intensity_threshold)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')

    def pred_2d_img(
            self,
            input_img_path: str,
            output_csv_path: str,
            enhanced_output_path: T.Optional[str] = None,
            connectivity: int = 2,
            intensity_threshold: float = 0.1,
            ):
        """Predict spots in a 2d image.

        Args:
            input_img_path: Path to the input image.
            output_csv_path: Path to the output csv file.
            enhanced_output_path: Path to the enhanced image.
            connectivity: The connectivity for finding local maxima.
            intensity_threshold: The threshold for the intensity.
        """
        from skimage.io import imread, imsave
        if not self._weights_loaded:
            self.load_weights()
        logger.info(f'Predicting {input_img_path}')
        img = imread(input_img_path)
        pred_df, enhanced = self._ufish.pred_2d(
            img,
            connectivity=connectivity,
            intensity_threshold=intensity_threshold,
            return_enhanced_img=True)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')
        if enhanced_output_path is not None:
            imsave(enhanced_output_path, enhanced)
            logger.info(f'Saved enhanced image to {enhanced_output_path}')

    def pred_2d_imgs(
            self,
            input_path: str,
            output_dir: str,
            data_base_dir: T.Optional[str] = None,
            img_glob: T.Optional[str] = None,
            save_enhanced_img: bool = True,
            connectivity: int = 2,
            intensity_threshold: float = 0.1,
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
            connectivity: The connectivity for finding local maxima.
            intensity_threshold: The threshold for the intensity.
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
            output_path = out_dir_path / (input_prefix + '.pred.csv')
            enhanced_img_path = out_dir_path / (input_prefix + '.enhanced.tif')
            if enhanced_img_path.exists():
                logger.info(
                    f'Enhanced image {enhanced_img_path} exists, ' +
                    'skipping enhancement.')
                self.call_spots_local_maxima(
                    str(enhanced_img_path),
                    str(output_path),
                    connectivity=connectivity,
                    intensity_threshold=intensity_threshold,
                )
            else:
                self.pred_2d_img(
                    str(in_path),
                    str(output_path),
                    str(enhanced_img_path) if save_enhanced_img else None,
                    connectivity=connectivity,
                    intensity_threshold=intensity_threshold,
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
            target_process: str = 'gaussian',
            data_argu: bool = False,
            num_epochs: int = 50,
            batch_size: int = 8,
            lr: float = 1e-3,
            summary_dir: str = "runs/unet",
            model_save_path: str = "best_unet_model.pth",
            only_save_best: bool = True,
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
                'gaussian' or 'dialation'. default 'gaussian'.
            data_argu: Whether to use data augmentation.
            num_epochs: The number of epochs to train.
            batch_size: The batch size.
            lr: The learning rate.
            summary_dir: The directory to save the TensorBoard summary to.
            model_save_path: The path to save the best model to.
        """
        self._ufish.train(
            train_path=train_path,
            valid_path=valid_path,
            root_dir=root_dir,
            img_glob=img_glob,
            coord_glob=coord_glob,
            target_process=target_process,
            data_argu=data_argu,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            summary_dir=summary_dir,
            model_save_path=model_save_path,
            only_save_best=only_save_best,
        )
