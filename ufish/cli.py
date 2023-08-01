import typing as T
from os.path import splitext
from pathlib import Path

from fire import Fire
import pandas as pd
from skimage.io import imread, imsave
from matplotlib import pyplot as plt

from .utils.log import logger


class UFishCLI():
    def __init__(
            self,
            cuda: bool = True,
            local_store_path: str = '~/.ufish/',
            weights_file_name: str = 'v1-for_benchmark.pth',
            ):
        from .api import UFish
        self._ufish = UFish(
            cuda=cuda,
            default_weight_file=weights_file_name,
            local_store_path=local_store_path,
        )
        self._weights_loaded = False

    def load_weights(
            self,
            weights_path: T.Optional[str] = None):
        """Load weights from a local file or the internet.

        Args:
            weights_path: The path to the weights file."""
        if weights_path is not None:
            self._ufish.load_weights(weights_path)
        else:
            self._ufish.load_weights_from_internet()
        self._weights_loaded = True

    def enhance_img(
            self,
            input_img_path: str,
            output_img_path: str,
            ):
        """Enhance an image."""
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
        img = imread(enhanced_img_path)
        logger.info(f'Calling spots in {enhanced_img_path}')
        pred_df = self._ufish.call_spots_cc_center(
            img,
            binary_threshold=binary_threshold,
            cc_size_thresh=cc_size_thresh)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')

    def pred_2d_img(
            self,
            input_img_path: str,
            output_csv_path: str,
            enhanced_output_path: T.Optional[str] = None,
            binary_threshold: T.Union[str, float] = 'otsu',
            cc_size_thresh: int = 20
            ):
        """Predict spots in a 2d image.

        Args:
            input_img_path: Path to the input image.
            output_csv_path: Path to the output csv file.
            enhanced_output_path: Path to the enhanced image.
            binary_threshold: The threshold for binarizing the image.
            cc_size_thresh: Connected component size threshold.
        """
        if not self._weights_loaded:
            self.load_weights()
        logger.info(f'Predicting {input_img_path}.')
        img = imread(input_img_path)
        pred_df, enhanced = self._ufish.pred_2d(
            img, binary_threshold=binary_threshold,
            cc_size_thresh=cc_size_thresh,
            return_enhanced_img=True)
        pred_df.to_csv(output_csv_path, index=False)
        logger.info(f'Saved predicted spots to {output_csv_path}')
        if enhanced_output_path is not None:
            imsave(enhanced_output_path, enhanced)
            logger.info(f'Saved enhanced image to {enhanced_output_path}')

    def pred_2d_imgs(
            self,
            input_dir: str,
            output_dir: str,
            save_enhanced_img: bool = True,
            cc_size_thresh: int = 18
            ):
        """Predict spots in a directory of 2d images.

        Args:
            input_dir: Path to the input directory.
            output_dir: Path to the output directory.
            save_enhanced_img: Whether to save the enhanced image.
            cc_size_thresh: Connected component size threshold.
        """
        if not self._weights_loaded:
            self.load_weights()
        in_dir_path = Path(input_dir)
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Predicting images in {in_dir_path}')
        logger.info(f'Saving results to {out_dir_path}')
        for input_path in in_dir_path.iterdir():
            input_prefix = splitext(input_path.name)[0]
            output_path = out_dir_path / (input_prefix + '.pred.csv')
            enhanced_img_path = out_dir_path / (input_prefix + '.enhanced.tif')
            self.pred_2d_img(
                str(input_path),
                str(output_path),
                str(enhanced_img_path) if save_enhanced_img else None,
                cc_size_thresh
            )

    def plot_2d_pred(
            self,
            image_path: str,
            pred_csv_path: str,
            fig_save_path: T.Optional[str] = None,
            **kwargs
            ):
        """Plot the predicted spots on the image."""
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


if __name__ == '__main__':
    Fire(UFishCLI())
