import numpy as np
from skimage.exposure import rescale_intensity


def scale_image(img: np.ndarray) -> np.ndarray:
    dtype = img.dtype
    img = img.astype(np.float32)
    if dtype is not np.uint8:
        img = rescale_intensity(img, out_range=(0, 255))
    return img
