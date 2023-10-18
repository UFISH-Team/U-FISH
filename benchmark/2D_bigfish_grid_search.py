""" 2D grid search script. """

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from glob import glob
import pandas as pd
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish

import time

import multiprocessing

print("Big-FISH version: {0}".format(bigfish.__version__), flush=True)

## Copied RS-FISH team some code to get all possible thresholds
def _get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.
    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].
    Parameters
    ----------
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.
    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.
    """
    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y

def _get_spot_counts(thresholds, value_spots):
    """Compute and format the spots count function for different thresholds.
    Parameters
    ----------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    value_spots : np.ndarray
        Pixel intensity values of all spots.
    Returns
    -------
    count_spots : np.ndarray, np.float64
        Spots count function.
    """
    # count spots for each threshold
    count_spots = np.log([np.count_nonzero(value_spots > t)
                          for t in thresholds])
    count_spots = stack.centered_moving_average(count_spots, n=5)

    # the tail of the curve unnecessarily flatten the slop
    count_spots = count_spots[count_spots > 2]
    thresholds = thresholds[:count_spots.size]

    return thresholds, count_spots

def spots_thresholding(image, mask_local_max, threshold,
                       remove_duplicate=True):
    """Filter detected spots and get coordinates of the remaining spots.
    In order to make the thresholding robust, it should be applied to a
    filtered image (bigfish.stack.log_filter for example). If the local
    maximum is not unique (it can happen with connected pixels with the same
    value), connected component algorithm is applied to keep only one
    coordinate per spot.
    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the spots.
    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])
    stack.check_parameter(threshold=(float, int),
                          remove_duplicate=bool)

    # remove peak with a low intensity
    mask = (mask_local_max & (image > threshold))
    if mask.sum() == 0:
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        return spots, mask

    # make sure we detect only one coordinate per spot
    if remove_duplicate:
        # when several pixels are assigned to the same spot, keep the centroid
        cc = label(mask)
        local_max_regions = regionprops(cc)
        spots = []
        for local_max_region in local_max_regions:
            spot = np.array(local_max_region.centroid)
            spots.append(spot)
        spots = np.stack(spots).astype(np.int64)

        # built mask again
        mask = np.zeros_like(mask)
        mask[spots[:, 0], spots[:, 1]] = True

    else:
        # get peak coordinates
        spots = np.nonzero(mask)
        spots = np.column_stack(spots)

    return spots, mask

def _get_candidate_thresholds(pixel_values):
    """Choose the candidate thresholds to test for the spot detection.
    Parameters
    ----------
    pixel_values : np.ndarray
        Pixel intensity values of the image.
    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    """
    # choose appropriate thresholds candidate
    start_range = 0
    end_range = int(np.percentile(pixel_values, 99.9999))
    if end_range < 100:
        thresholds = np.linspace(start_range, end_range, num=100)
    else:
        thresholds = [i for i in range(start_range, end_range + 1)]
    thresholds = np.array(thresholds)
  
    return thresholds
  
def automated_threshold_setting(image, mask_local_max):
    """Automatically set the optimal threshold to detect spots.
    In order to make the thresholding robust, it should be applied to a
    filtered image (bigfish.stack.log_filter for example). The optimal
    threshold is selected based on the spots distribution. The latter should
    have a kink discriminating a fast decreasing stage from a more stable one
    (a plateau).
    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    Returns
    -------
    optimal_threshold : int
        Optimal threshold to discriminate spots from noisy blobs.
    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(image.ravel())

    # get spots count and its logarithm
    first_threshold = float(thresholds[0])
    spots, mask_spots = spots_thresholding(
        image, mask_local_max, first_threshold, remove_duplicate=False)
    value_spots = image[mask_spots]
    thresholds, count_spots = _get_spot_counts(thresholds, value_spots)

    # select threshold where the kink of the distribution is located
    optimal_threshold, _, _ = _get_breaking_point(thresholds, count_spots)

    return thresholds.astype(float), optimal_threshold


def process_im(im, sigma, voxel_size_yx, psf_yx, im_path, thr_range, gamma_range, alpha_range, beta_range):

    ## Filter image LoG:
    try:
        rna_log = stack.log_filter(im, sigma)
    except Exception as e:
        print(f"Error in log_filter for {im_path}: {str(e)}")
        return  # Go back immediately and do not continue working on the image

    ## Detect local maxima:
    try:
        mask = detection.local_maximum_detection(rna_log, min_distance=sigma)
    except Exception as e:
        print(f"Error in local_maximum_detection for {im_path}: {str(e)}")
        return  # Go back immediately and do not continue working on the image

    ## Find defualt threshold + all thresholds:
    all_thrs, default_thr = automated_threshold_setting(rna_log, mask)

    # Find thresholds to test around the default
    idx_default_thr = np.where(all_thrs == default_thr)[0][0]
    thrs_to_use_idxs = [it + idx_default_thr for it in thr_range if 0 < (it + idx_default_thr) < all_thrs.size]
    thrs_to_use = all_thrs[thrs_to_use_idxs]

    ## Iterate thresholds

    # So we won't run the same spots found - if two or more thresholds have same # of spots
    # Only one is ran.
    n_spots = []
    ## GRID PARAMETER
    for threshold in thrs_to_use:

        ## Detect spots
        try:
            spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
        except Exception as e:
            print(f"Error in spots_thresholding for {im_path} with threshold {threshold}: {str(e)}")
            continue  # Continue processing for the next threshold

        thr_n_spots = spots.shape[0]

        if (thr_n_spots not in n_spots):
            n_spots.append(thr_n_spots)

            conditions_str = (f'BF_{os.path.basename(im_path)}'
                          f'_sigyx{sigma[0]}thr{threshold:.3f}')

            print(f'sigma={sigma} threshold={threshold:.3f}, detected #spots={spots.shape[0]}', flush=True)

            ## save the spots (int)
            df = pd.DataFrame(data=spots, columns=["axis-0", "axis-1"])
            df_path = os.path.join(os.path.dirname(csv_path),
                                    (f'{conditions_str}_direct_spots.csv'))
            df.to_csv(df_path, index=False)

        ### Dense region decomposition:

        ## GRID PARAMETER
        for gamma in gamma_range:
            ## GRID PARAMETER
            for alpha in alpha_range:
                ## GRID PARAMETER
                for beta in beta_range:
                    time_tmp2 = time.time()
                    print(f'starting dense regions decomposition with alpha {alpha}, beta {beta}, gamma {gamma}', flush=True)

                    try:
                        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                            im, spots, voxel_size_yx, psf_yx,
                            alpha=alpha,  # alpha impacts the number of spots per candidate region
                            beta=beta,  # beta impacts the number of candidate regions to decompose
                            gamma=gamma)  # gamma the filtering step to denoise the image
                        if spots_post_decomposition.shape[0] < thr_n_spots:
                            print('#spots after decomposition < #spots before. Not saving results.')
                        else:
                            df = pd.DataFrame(data=spots_post_decomposition, columns=["axis-0", "axis-1"])
                            df_path = os.path.join(os.path.dirname(csv_path),
                                                   (f'{conditions_str}_alpha{alpha}_beta{beta}_gamma{gamma}_spots_decomposition.csv'))
                            df.to_csv(df_path, index=False)
                            time2 = int(round((time.time() - time_tmp2) * 1000))
                            print(f'saving to {df_path}')
                            print(f'running single image time: {time2}')
                    except Exception as e:
                        print(f"Error in decompose_dense for {im_path} with alpha {alpha}, beta {beta}, gamma {gamma}: {str(e)}")
                        continue  # Continue processing of the next parameter combination


## Process single image:
def process_image(im_path):

    ## as threshold depands on the default threshold,
    # we just choose locations in the threshold array in referance to the default
    # The default is the first, so it will be ran for sure:
    thr_range = [0, -6, -3, -2, -1, 1, 2, 3, 6]

    im = stack.read_image(im_path)
    print(f'processing image: {im_path}', flush=True)
    ## GRID PARAMETER
    for sig_yx_delta in sigma_yx_range:
        sigma_yx = sig_yx_delta + default_sig_yx

        psf_yx = sigma_yx * voxel_size_yx

        sigma = (sigma_yx, sigma_yx)

        process_im(im, sigma, voxel_size_yx, psf_yx,
                    im_path, thr_range, gamma_range, alpha_range, beta_range)

time_tmp1 = time.time()

#path = '/path/to/input/'
path = './valid/'
ims_path = 'image/'

# path = '/path/to/output/
csv_path = './valid/BF_results/'

## Set parameters:
sigma_yx_range = np.arange(-0.5,0.51,0.25)

##### Dense region decomposition parameters:

## Gamma - used for denoising: large_sigma = tuple([sigma_ * gamma for sigma_ in sigma])
## Five is default
gamma_range = [4, 4.5, 5, 5.5, 6.1]
alpha_range = np.arange(0.5,0.8,0.1)
beta_range = np.arange(0.8,1.3,0.1)

# set voxel size:
voxel_size_yx = 1
default_sig_yx = 1.5

# Gets a list of image files to process:
image_files = glob(os.path.join(path, ims_path, "*.tif"))

## Create a process pool
#num_processes = multiprocessing.cpu_count()
num_processes = 100
pool = multiprocessing.Pool(num_processes)

## Process images in parallel using process pools
pool.map(process_image, image_files)

## Close the process pool
pool.close()
pool.join()

time1 = int(round((time.time() - time_tmp1)*1000))
print(f'running all images time: {time1}')
