import typing as T
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity


def scale_image(
        img: np.ndarray,
        big_quantile: float = 0.9999,
        warning: bool = False,
        ) -> np.ndarray:
    """Scale an image to 0-255.
    If the image has outlier values,
    the image will be scaled to 0-big_value.

    Args:
        img: Image to scale.
        big_quantile: Quantile to calculate the big value.
        warning: Whether to print a warning message.
    """
    dtype = img.dtype
    img = img.astype(np.float32)
    if dtype is not np.uint8:
        big_value = np.quantile(img, big_quantile)
        if img_has_outlier(img, big_value):
            if warning:
                from .log import logger
                logger.warning(
                    'Image has outlier values. ')
            in_range = (0, big_value)
        else:
            in_range = 'image'
        img = rescale_intensity(
            img,
            in_range=in_range,
            out_range=(0, 255),
        )
    return img


def img_has_outlier(
        img: np.ndarray,
        big_value: float,
        ) -> bool:
    """Check if an image has outlier values.
    If the difference between the maximum value
    and the big value is greater than the big value,
    then the image has outlier values.

    Args:
        img: Image to check.
        big_value: Value to compare with the maximum value.
    """
    max_value = np.max(img)
    diff = max_value - big_value
    if diff > big_value:
        return True
    else:
        return False


def infer_img_axes(shape: tuple) -> str:
    """Infer the axes of an image.

    Args:
        shape: Shape of the image.
    """
    if len(shape) == 2:
        return 'yx'
    elif len(shape) == 3:
        return 'zyx'
    elif len(shape) == 4:
        return 'czyx'
    elif len(shape) == 5:
        return 'tczyx'
    else:
        raise ValueError(
            f'Image shape {shape} is not supported. ')


def check_img_axes(axes: str):
    """Check if the axes of an image is valid.

    Args:
        axes: Axes of the image.
    """
    if len(axes) < 2 or len(axes) > 5:
        raise ValueError(
            f'Axes {axes} is not supported. ')
    if len(axes) != len(set(axes)):
        raise ValueError(
            f'Axes {axes} must be unique. ')
    if 'y' not in axes:
        raise ValueError(
            f'Axes {axes} must contain y. ')
    if 'x' not in axes:
        raise ValueError(
            f'Axes {axes} must contain x. ')


def expand_df_axes(
        df: pd.DataFrame, axes: str,
        axes_vals: T.Sequence[int],
        ) -> pd.DataFrame:
    """Expand the axes of a DataFrame."""
    # insert new columns
    for i, vals in enumerate(axes_vals):
        df.insert(i, axes[i], vals)
    df.columns = list(axes)
    return df


def map_predfunc_to_img(
        predfunc: T.Callable[
            [np.ndarray],
            T.Tuple[pd.DataFrame, np.ndarray]
        ],
        img: np.ndarray,
        axes: str,
        inplace: bool = False,
        ):
    from .log import logger
    yx_idx = [axes.index(c) for c in 'yx']
    # move yx to the last two axes
    img = np.moveaxis(img, yx_idx, [-2, -1])
    df_axes = axes.replace('y', '').replace('x', '') + 'yx'
    dfs = []
    if inplace:
        e_img = img
        e_img = e_img.astype(np.float32)
    if len(img.shape) in (2, 3):
        df, e_img = predfunc(img)
        df = expand_df_axes(df, df_axes, [])
        dfs.append(df)
    elif len(img.shape) == 4:
        e_img = np.zeros_like(img)
        for i, img_3d in enumerate(img):
            logger.log(f'Processing image {i+1}/{len(img)}')
            df, e_img[i] = predfunc(img_3d)
            df = expand_df_axes(df, df_axes, [i])
            dfs.append(df)
    else:
        assert len(img.shape) == 5
        e_img = np.zeros_like(img)
        num_imgs = img.shape[0] * img.shape[1]
        for i, img_4d in enumerate(img):
            for j, img_3d in enumerate(img_4d):
                logger.log(
                    f'Processing image {i*img.shape[1]+j+1}/{num_imgs}')
                df, e_img[i, j] = predfunc(img_3d)
                df = expand_df_axes(df, df_axes, [i, j])
                dfs.append(df)
    # move yx back to the original position
    e_img = np.moveaxis(e_img, [-2, -1], yx_idx)
    res_df = pd.concat(dfs, ignore_index=True)
    # re-order columns
    res_df = res_df[list(axes)]
    res_df.columns = [f'axis-{i}' for i in range(len(axes))]
    return res_df, e_img
