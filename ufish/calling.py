"""
Spots calling from the enhanced images.
"""
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def call_dense_region(binary_image: np.ndarray) -> list:
    distance = ndi.distance_transform_edt(binary_image)
    coords = peak_local_max(
        distance,
        footprint=np.ones((4, 4)),
        labels=binary_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_image)
    regions = regionprops(label(labels))
    centroids = [r.centroid for r in regions]
    return centroids


def call_spots(
        image: np.ndarray,
        cc_size_threshold: int = 18,
        output_dense_mark: bool = False,
        ) -> pd.DataFrame:
    ndim = image.ndim
    binary_image = image > threshold_otsu(image)
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
    centroids_dense = call_dense_region(dense_regions)
    all_centroids = centroids_sparse + centroids_dense
    all_centroids = np.array(all_centroids)  # type: ignore
    columns = [f'axis-{i}' for i in range(ndim)]
    df = pd.DataFrame(
        all_centroids,
        columns=columns
    )
    if output_dense_mark:
        dense_mark = np.zeros(len(all_centroids), dtype=bool)
        dense_mark[len(centroids_sparse):] = True
        df['is_dense'] = dense_mark
    return df
