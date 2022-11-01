import argparse
from glob import glob
from os import path

from natsort import natsorted
from tqdm import tqdm

import imageio
import numpy as np
from skimage import img_as_float32

assets_dir = path.abspath(path.join(path.dirname(__file__), '..', 'assets'))

default_input_size = [768, 1024]
default_input_dir = path.join(
    assets_dir,
    f'puzzle_corners_{default_input_size[1]}x{default_input_size[0]}')


def default_parser(description):
    parser = argparse.ArgumentParser(description=description)

    # Set device
    parser.add_argument('--cpu',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Run in CPU')

    # Random seed
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random number seed')

    # Input directory
    parser.add_argument('--input-dir',
                        default=default_input_dir,
                        help='path to image and mask dirs')

    # Input size, builds out path
    parser.add_argument('--input-size',
                        default=default_input_size,
                        nargs=2,
                        type=int,
                        metavar=('height', 'width'))

    return parser


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


def split_data(raw_images, raw_masks):
    N = len(raw_images)
    indices = np.random.permutation(N)

    train_end = int(0.7*N) + 1
    train_indices = indices[:train_end]

    validation_end = train_end + int(0.15*N)
    validation_indices = indices[train_end:validation_end]

    test_indices = indices[validation_end:]

    return (raw_images[train_indices], raw_masks[train_indices]),\
        (raw_images[validation_indices], raw_masks[validation_indices]),\
        (raw_images[test_indices], raw_masks[test_indices])
