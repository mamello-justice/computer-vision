import argparse

import common
from gmm.gmm import GMM


def setup_args():
    parser = common.default_parser(
        description=f'Gaussian Mixture Model (GMM)')

    # GMM for foreground or background?
    gmm_type = parser.add_mutually_exclusive_group(required=True)
    gmm_type.add_argument('--foreground', '-fg',
                          action=argparse.BooleanOptionalAction,
                          help='train GMM for foreground pixels')
    gmm_type.add_argument('--background', '-bg',
                          action=argparse.BooleanOptionalAction,
                          help='train GMM for background pixels')

    # Feature group
    feature_group = parser.add_argument_group('features')
    feature_group.add_argument('--rgb',
                               action=argparse.BooleanOptionalAction,
                               default=True,
                               help='use RGB values in feature set')

    feature_group.add_argument('--diff-gauss', '-DoG',
                               action=argparse.BooleanOptionalAction,
                               default=False,
                               help='use Difference of Gaussian in feature set')

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = setup_args()

    input_dir = args['input_dir']
    input_size = args['input_size']

    height, width = input_size

    raw_images, raw_masks = common.load_images(input_dir, height, width)

    foreground = args['foreground']
    background = args['background']

    # Features
    rgb = args['rgb']
    diff_gauss = args['diff_gauss']

    enabled_features = {
        rgb: rgb,
        diff_gauss: diff_gauss,
    }

    if foreground is not None and foreground:
        print("train foreground")
        gmm = GMM(enabled_features)

    elif background is not None and background:
        print("train background")
        gmm = GMM(enabled_features)
