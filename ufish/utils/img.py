import typing as T
import numpy as np
import pandas as pd
from itertools import product
from os.path import isdir

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
        min_dim_idx = shape.index(min(shape))
        low_dim_shape = list(shape)
        low_dim_shape.pop(min_dim_idx)
        low_dim_axes = infer_img_axes(tuple(low_dim_shape))
        return low_dim_axes[:min_dim_idx] + 'z' + low_dim_axes[min_dim_idx:]
    elif len(shape) == 4:
        min_dim_idx = shape.index(min(shape))
        low_dim_shape = list(shape)
        low_dim_shape.pop(min_dim_idx)
        low_dim_axes = infer_img_axes(tuple(low_dim_shape))
        return low_dim_axes[:min_dim_idx] + 'c' + low_dim_axes[min_dim_idx:]
    elif len(shape) == 5:
        low_dim_shape = infer_img_axes(shape[1:])
        return 't' + low_dim_shape
    else:
        raise ValueError(
            f'Image shape {shape} is not supported. ')


def check_img_axes(img: np.ndarray, axes: str):
    """Check if the axes of an image is valid.

    Args:
        img: Image to check.
        axes: Axes of the image.
    """
    if len(img.shape) != len(axes):
        raise ValueError(
            f'Axes {axes} does not match image shape {img.shape}. ')
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
        ):
    from .log import logger
    yx_idx = [axes.index(c) for c in 'yx']
    # move yx to the last two axes
    img = np.moveaxis(img, yx_idx, [-2, -1])
    new_axes = axes.replace('y', '').replace('x', '') + 'yx'
    dfs = []
    if len(img.shape) in (2, 3):
        df, e_img = predfunc(img, axes=axes)
        df = expand_df_axes(df, new_axes, [])
        dfs.append(df)
    elif len(img.shape) == 4:
        e_img = np.zeros_like(img, dtype=np.float32)
        for i, img_3d in enumerate(img):
            logger.info(
                'Processing multi-dimensional image'
                f' {i+1}/{len(img)}')
            df, e_img[i] = predfunc(img_3d, axes=axes[1:])
            df = expand_df_axes(df, new_axes, [i])
            dfs.append(df)
    else:
        assert len(img.shape) == 5
        e_img = np.zeros_like(img, dtype=np.float32)
        num_imgs = img.shape[0] * img.shape[1]
        for i, img_4d in enumerate(img):
            for j, img_3d in enumerate(img_4d):
                logger.info(
                    'Processing multi-dimensional image'
                    f' {i*img.shape[1]+j+1}/{num_imgs}'
                    )
                df, e_img[i, j] = predfunc(img_3d, axes=axes[2:])
                df = expand_df_axes(df, new_axes, [i, j])
                dfs.append(df)
    # move yx back to the original position
    e_img = np.moveaxis(e_img, [-2, -1], yx_idx)
    res_df = pd.concat(dfs, ignore_index=True)
    # re-order columns
    res_df = res_df[list(axes)]
    res_df.columns = [f'axis-{i}' for i in range(len(axes))]
    return res_df, e_img


def get_default_chunk_size(
        axes: str,
        default_x: T.Union[int, str] = 512,
        default_y: T.Union[int, str] = 512,
        default_z: T.Union[int, str] = 'image',
        default_c: T.Union[int, str] = 'image',
        default_t: T.Union[int, str] = 'image',
        ) -> tuple:
    """Get the default chunk size of an image.

    Args:
        img_shape: Shape of the image.
        axes: Axes of the image.
        default_x: Default chunk size for x axis.
            'image' means the whole image.
        default_y: Default chunk size for y axis.
            'image' means the whole image.
        default_z: Default chunk size for z axis.
            'image' means the whole image.
        default_c: Default chunk size for c axis.
            'image' means the whole image.
        default_t: Default chunk size for t axis.
            'image' means the whole image.
    """
    default_sizes = {
        'y': default_y,
        'x': default_x,
        'z': default_z,
        'c': default_c,
        't': default_t,
    }
    chunk_size = []

    for c in axes:
        if c in default_sizes:
            chunk_size.append(default_sizes[c])
        else:
            raise ValueError(
                f'Axis {c} is not supported. ')
    return tuple(chunk_size)


def process_chunk_size(
        chunk_size: T.Tuple[T.Union[int, str], ...],
        img_shape: T.Tuple[int, ...],
        ) -> T.Tuple[int, ...]:
    """Process the chunk size of an image.
    If the chunk size is 'image', then the chunk size
    will be the same as the image shape.

    Args:
        chunk_size: Chunk size of the image.
        img_shape: Shape of the image.
    """
    assert len(chunk_size) == len(img_shape), \
        "chunk_size and img_shape must have the same length"
    new_chunk_size = []
    for i, size in enumerate(chunk_size):
        if size == 'image':
            new_chunk_size.append(img_shape[i])
        else:
            new_chunk_size.append(size)
    return tuple(new_chunk_size)


def get_chunks_range(
        img_shape: tuple,
        chunk_size: tuple,
        ) -> T.List[T.List[T.List[int]]]:
    """Get the ranges of each chunk.
    For example, if the image shape is (100, 100)
    and the chunk size is (50, 50), then the ranges
    of the chunks are: [
        [[0, 50], [0, 50]],
        [[0, 50], [50, 100]],
        [[50, 100], [0, 50]],
        [[50, 100], [50, 100]],
    ]

    Args:
        img_shape: Shape of the image.
        chunk_size: Chunk size of the image.
    """
    ranges_each_dim = []
    for dim_size, chunk_dim_size in zip(img_shape, chunk_size):
        num_chunks = int(np.ceil(dim_size / chunk_dim_size))
        dim_ranges = []
        for j in range(num_chunks):
            start = j * chunk_dim_size
            end = min((j + 1) * chunk_dim_size, dim_size)
            dim_ranges.append([start, end])
        ranges_each_dim.append(dim_ranges)
    chunk_ranges = list(product(*ranges_each_dim))
    return chunk_ranges


def chunks_iterator(
        original_img: np.ndarray,
        chunk_size: tuple,
        padding: bool = True,
        ) -> T.Iterator[
            T.Tuple[T.List[T.List[int]],
                    np.ndarray]]:
    """Iterate over chunks of an image.

    Args:
        original_img: Image to iterate over.
        chunk_size: Chunk size of the image.
    """
    chunk_ranges = get_chunks_range(
        original_img.shape, chunk_size)
    for chunk_range in chunk_ranges:
        chunk = original_img[
            tuple(slice(*r) for r in chunk_range)]
        if padding:
            chunk = np.pad(
                chunk,
                [(0, chunk_size[i] - (r[1] - r[0]))
                 for i, r in enumerate(chunk_range)],
                mode='constant',
                constant_values=0,
            )
        yield chunk_range, chunk


def enhance_blend_3d(
        img: np.ndarray,
        enh_func: T.Callable[[np.ndarray], np.ndarray],
        axes: str,
        ) -> np.ndarray:
    """Run enhancement along 3 directions and blend the results.

    Args:
        enh_func: Enhancement function.
        img: Image to enhance.
        axes: Axes of the image.
    """
    if axes != 'zyx':
        # move z to the first axis
        z_idx = axes.index('z')
        img = np.moveaxis(img, z_idx, 0)
    enh_z = enh_func(img)
    enh_y = enh_func(np.moveaxis(img, 1, 0))
    enh_y = np.moveaxis(enh_y, 0, 1)
    enh_x = enh_func(np.moveaxis(img, 2, 0))
    enh_x = np.moveaxis(enh_x, 0, 2)
    enh_img = enh_z * enh_y * enh_x
    return enh_img


def open_for_read(path: str):
    from .ngff import is_ngff_suffix
    if is_ngff_suffix(path) or isdir(path):
        from .ngff import read_ngff
        img = read_ngff(path)
    elif path.endswith('.zarr'):
        import zarr
        img = zarr.open(path, 'r')
    elif path.endswith('.n5'):
        import zarr
        store = zarr.N5Store(path)
        img = zarr.open(store, 'r')
    else:
        from skimage.io import imread
        img = imread(path)
    return img


def open_for_write(
        path: str, shape: tuple,
        dtype=np.float32):
    img = None
    if path is not None:
        from .ngff import is_ngff_suffix
        if path.endswith('.zarr') or is_ngff_suffix(path):
            import zarr
            img = zarr.open(
                path, 'w', shape=shape, dtype=dtype)
        elif path.endswith('.n5'):
            import zarr
            store = zarr.N5Store(path)
            img = zarr.zeros(
                shape, dtype=dtype,
                store=store, overwrite=True)
    return img


def open_enhimg_storage(enh_path: str, shape: tuple):
    from .ngff import is_ngff_suffix
    tmp_enh_path = None
    if (enh_path is not None) and is_ngff_suffix(enh_path):
        if enh_path.endswith("/"):
            enh_path = enh_path[:-1]
        tmp_enh_path = enh_path + '.tmp.zarr'
        enhanced = open_for_write(tmp_enh_path, shape)
    else:
        enhanced = open_for_write(enh_path, shape)
    return enhanced, tmp_enh_path


def save_enhimg(
        enhanced, tmp_enh_path,
        enh_path: str, axes: str):
    from .ngff import is_ngff_suffix
    from .log import logger
    if is_ngff_suffix(enh_path):
        logger.info("Saving enhanced image to ngff file.")
        from .ngff import create_ngff, generate_omero_info
        omero_info = generate_omero_info(data=enhanced, axes=axes)
        create_ngff(
            data=enhanced, out_path=enh_path,
            axes=axes, omero_info=omero_info)
        if tmp_enh_path is not None:
            import shutil
            shutil.rmtree(tmp_enh_path)
    elif enh_path.endswith('.zarr'):
        logger.info("Saving enhanced image to zarr file.")
    elif enh_path.endswith('.n5'):
        logger.info("Saving enhanced image to n5 file.")
    else:
        from skimage.io import imsave
        imsave(enh_path, enhanced, check_contrast=False)
    logger.info(f'Saved enhanced image to {enh_path}')
