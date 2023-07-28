import os
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.io import imread

def call_dense_region(binary_image):

    distance = ndi.distance_transform_edt(binary_image)
    coords = peak_local_max(distance, footprint=np.ones((4, 4)), labels=binary_image)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask=binary_image)
    regions = regionprops(label(labels))
    centroids = [r.centroid for r in regions]

    return centroids


def call_spots(image): 
    binary_image = image > threshold_otsu(image)
    
    regions = regionprops(label(binary_image))
    signal_threshold = 18
    centroids_sparse = [r.centroid for r in regions if r.area <= signal_threshold]
    dense_regions = np.zeros_like(binary_image, dtype=np.uint8)

    for region in regions:
        if region.area > signal_threshold:
            dense_regions[region.coords[:, 0], region.coords[:, 1]] = 1

    centroids_dense = call_dense_region(dense_regions)
    all_centroids = centroids_sparse + centroids_dense

    return  pd.DataFrame(all_centroids, columns= ['axis=0', 'axis-1'])   # loading csv format in napari


# single image
def call_image(img_path): 
    image = imread(img_path)
    df = call_spots(image)
    csv_path = img_path.replace(".tif", ".csv")
    df.to_csv(csv_path, index = False)


# Multiple images
def call_images(img_path): 
    for img in os.listdir(img_path):
        image = imread(os.path.join(img_path,img))
        csv = img.replace(".tif", ".csv")
        df = call_spots(image)
        df.to_csv(os.path.join(img_path,csv), index = False)

  
