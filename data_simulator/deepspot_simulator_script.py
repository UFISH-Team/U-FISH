# -*- coding: utf-8 -*-
# /usr/bin/python3.8

import os
import pathlib
import random
from random import randrange

import cv2
import elasticdeform
import numpy as np
import scipy.stats as st
from scipy import signal
from skimage import exposure
from tqdm import tqdm



def rotate_img(path, img_name):
    img = cv2.imread(path + img_name, cv2.IMREAD_UNCHANGED)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    angle = 90
    scale = 1.0

    # Perform the counter clockwise rotation holding at the center

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (h, w))
    return rotated


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise


def random_unispots_coordinates(width, height):
    arr = np.zeros((width, height))
    x = randrange(2, width - 2)
    y = randrange(2, height - 2)
    arr[x][y] = 1

    return arr


def gaussian_kernel(kernel_size, std):
    """Returns a 2D Gaussian kernel."""
    # ---- Create a 1D kernel of size kernel_size between values -std and std
    x = np.linspace(-std, std, kernel_size + 1)
    # ---- Calculate the differecence between subsequent values over the umulative distribution
    kernel_1d = np.diff(st.norm.cdf(x))
    # --- Goes 2D by the product of the kernel 1D by himself
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    # normalize and return
    return kernel_2d / kernel_2d.sum()


def apply_kernel(img, x, y, kernel):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (x + i) < 256 and (y + j) < 256:
                img[x + i][y + j] = kernel[i + 1][j + 1]

    return img


if __name__ == '__main__':

    root = "var_intensities/"
    spot_number = random.randint (50,400)
    kernel = np.array([[200, 230, 200], [230, 255, 230], [200, 230, 200]])
    out_image_size = 512
    image_number = 200


    pathlib.Path(os.path.join(root, 'original')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(root, 'target')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(root, 'csv')).mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(1, image_number+1)):
        spot_size = random.randint(4, 8)  # The bigger X, the smaller the spot
        kernel_size = random.randint(13, 18)
        perlin_level = random.choice([2, 4, 8, 16])
        poisson_noise_level = random.randint(50, 100)
        perlin_out_range = random.randint(50, 100)
        full_noise_level = random.randint(80, 150)

        coord_map = random_unispots_coordinates(out_image_size, out_image_size)
        psf_kernel = gaussian_kernel(kernel_size, std=spot_size)
        img = signal.convolve2d(coord_map, psf_kernel, mode="same")

        for i in range(0, spot_number):
            spot_size = random.randint(4, 8)
            kernel_size = random.randint(13, 18)
            coord_map2 = random_unispots_coordinates(out_image_size, out_image_size)
            psf_kernel = gaussian_kernel(kernel_size, std=spot_size)
            img2 = signal.convolve2d(coord_map2, psf_kernel, mode="same")
            img = img + img2
            coord_map = coord_map + coord_map2

        img = exposure.rescale_intensity(img, out_range=(0, random.randint(160, 220)))
        spot_coords = np.where(coord_map == 1)

        #### Generate poisson noise
        noise_map = np.random.normal(loc=0, scale=1, size=(1024, 1024))
        poisson_noise = exposure.rescale_intensity(noise_map, out_range=(0, poisson_noise_level))

        #### Generate perlin noise
        perlin_noise = generate_fractal_noise_2d((1024, 1024), (perlin_level, perlin_level), 5)
        perlin_noise = exposure.rescale_intensity(perlin_noise, out_range=(0, perlin_out_range))

        #### Normalize noise
        sum_noise = perlin_noise + poisson_noise
        norm_sum_noise = exposure.rescale_intensity(sum_noise, out_range=(0, full_noise_level))

        #### Apply elastic transform on noise
        cmin = int(norm_sum_noise.shape[0] / 2 - out_image_size / 2)
        cmax = int(norm_sum_noise.shape[0] / 2 + out_image_size / 2)
        crop = (slice(cmin, cmax), slice(cmin, cmax))
        noise_elastic = elasticdeform.deform_random_grid(norm_sum_noise, sigma=random.randint(10, 20),
                                                         points=random.randint(2, 6), crop=crop)
        full_noise = exposure.rescale_intensity(noise_elastic, out_range=(0, full_noise_level))

        target = img.copy()
        img += full_noise
        img = exposure.rescale_intensity(img, out_range=(0, 511))

        target_slim = img.copy()
        all_coords = {}
        mask = np.zeros((out_image_size, out_image_size))

        for i in range(0, len(spot_coords[0])):
            x, y = spot_coords[1][i], spot_coords[0][i]
            pos = (x, y)
            all_coords[i] = {"x": x, "y": y}
            target_slim = apply_kernel(target_slim, y, x, kernel)
            cv2.circle(mask, pos, 2, (511, 511, 511), -1)

        cv2.imwrite(os.path.join(root, "original", str(idx+300) + ".png"), img)
        cv2.imwrite(os.path.join(root, "target", str(idx+300) + ".png"), target_slim)

        with open(os.path.join(root, "csv", str(idx+300) + '.csv'), 'w') as f:
            for key in all_coords.keys():
                f.write("%s,%s\n" % (all_coords[key]["x"], all_coords[key]["y"]))
