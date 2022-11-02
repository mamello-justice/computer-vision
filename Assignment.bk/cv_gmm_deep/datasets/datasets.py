from glob import glob
from os import path

import numpy as np

import imageio
from skimage import img_as_float32
from natsort import natsorted
from tqdm.auto import tqdm


def load_images(input_dir, height, width):
    image_dir = path.join(input_dir, f'images-{width}x{height}')
    image_paths = natsorted(glob(f'{image_dir}/*.png'))
    raw_images = np.array([
        img_as_float32(imageio.imread(path))
        for path in tqdm(image_paths, 'Reading in images')
    ])

    mask_dir = path.join(input_dir, f'masks-{width}x{height}')
    mask_paths = natsorted(glob(f'{mask_dir}/*.png'))
    raw_masks = np.array([
        img_as_float32(imageio.imread(path))
        for path in tqdm(mask_paths, 'Reading in masks')
    ])

    return raw_images, raw_masks
