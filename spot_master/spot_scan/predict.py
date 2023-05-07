
import pandas as pd
import numpy as np
from PIL import Image
from skimage.morphology import white_tophat
from skimage.feature import peak_local_max


def local_max(img, q):
    t = white_tophat(img)
    spots = np.array(peak_local_max(t, threshold_abs=q, min_distance=2))
    df = pd.DataFrame(spots)
    return df

def create_postive_patches(img, df, patch_size =5):
    '''
    img: np.array(from .tif)
    df: coordinates(from .csv)
    patch_size: tuple, size of the patches to extract
    '''
    pos_patches = []
    height, width = img.shape  
    for index, row in df.iterrows():
        y, x = row[0], row[1]                    
        if x < patch_size // 2 or y < patch_size // 2 or x > width - patch_size // 2 - 1 or y > height - patch_size // 2 - 1:           
            top, bottom = max(patch_size // 2 - y, 0), max(y - height + patch_size // 2 + 1, 0)
            left, right = max(patch_size // 2 - x, 0), max(x - width + patch_size // 2 + 1, 0)
            patch = np.pad(img[max(y-patch_size // 2, 0):y+patch_size // 2+1, max(x-patch_size // 2, 0):x+patch_size // 2+1], ((top, bottom), (left,
            right)), mode='constant')
            patch = patch.ravel()
        else:  
            patch = img[y-patch_size // 2:y+patch_size // 2+1, x-patch_size // 2:x+patch_size // 2+1]
            patch = patch.ravel()
            
        pos_patches.append([patch,y,x])
        
    return pos_patches

