"""
Spots calling from the enhanced images.
"""
import typing as T

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import local_maxima
from skimage.segmentation import watershed


def watershed_center(binary_image: np.ndarray) -> list:
    """Segment the dense regions using watershed algorithm.
    and return the centroids of the regions."""
    distance = ndi.distance_transform_edt(binary_image)
    ndim = binary_image.ndim
    if ndim == 2:
        coords = peak_local_max(
            distance,
            footprint=np.ones((4, 4)),
            labels=binary_image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
    elif ndim == 3:
        mask = local_maxima(distance)
    else:
        raise ValueError('Only 2D and 3D images are supported.')
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_image)
    regions = regionprops(label(labels))
    centroids = [r.centroid for r in regions]
    return centroids


def call_spots_cc_center(
        image: np.ndarray,
        binary_threshold: T.Union[str, float] = 'otsu',
        cc_size_threshold: int = 20,
        output_dense_mark: bool = False,
        ) -> pd.DataFrame:
    """Call spots from the connected components' centroids.

    Args:
        image: The input image.
        binary_threshold: The threshold for binarizing the image.
        cc_size_threshold: The threshold for connected components' size.
        output_dense_mark: Whether to output a column indicating whether
            the spot is from a dense region.
    """
    ndim = image.ndim
    if binary_threshold == 'otsu':
        thresh = threshold_otsu(image)
    else:
        thresh = binary_threshold
    binary_image = image > thresh
    regions = regionprops(label(binary_image))
    centroids_sparse = []
    dense_regions = np.zeros_like(binary_image, dtype=np.uint8)
    for region in regions:
        if region.area > cc_size_threshold:
            coords = region.coords
            if ndim == 2:
                dense_regions[coords[:, 0], coords[:, 1]] = 1
            elif ndim == 3:
                dense_regions[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            else:
                raise ValueError('Only 2D and 3D images are supported.')
        else:
            centroids_sparse.append(region.centroid)
    centroids_dense = watershed_center(dense_regions)
    all_centroids = centroids_sparse + centroids_dense
    all_centroids = np.array(all_centroids)  # type: ignore
    columns = [f'axis-{i}' for i in range(ndim)]
    if all_centroids.shape[0] == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(
            all_centroids,
            columns=columns
        )
        if output_dense_mark:
            dense_mark = np.zeros(len(all_centroids), dtype=bool)
            dense_mark[len(centroids_sparse):] = True
            df['is_dense'] = dense_mark
    return df


def call_spots_local_maxima(
        enhanced_img: np.ndarray,
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
    mask = local_maxima(enhanced_img, connectivity=connectivity)
    mask = mask & (enhanced_img > intensity_threshold)
    peaks = np.array(np.where(mask)).T
    df = pd.DataFrame(
        peaks, columns=[f'axis-{i}' for i in range(mask.ndim)])
    return df
