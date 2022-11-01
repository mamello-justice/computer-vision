import os
from os import path

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

import common

BACKBONE = 'vgg16'


def setup_args():
    parser = common.default_parser(description=f'Deep Learning (UNet')

    parser.add_argument('model_path',
                        help='path to saved model files')

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = setup_args()

    use_cpu = args['cpu']
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_dir = args['input_dir']
    input_size = args['input_size']
    model_path = args['model_path']
    model_path = path.abspath(model_path)

    height, width = input_size

    raw_images, raw_masks = common.load_images(input_dir, height, width)

    masks = np.expand_dims(raw_masks, axis=3)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) =\
        common.split_data(raw_images, masks)

    model = load_model(model_path)

    model.evaluate(train_x[:7], train_y[:7], batch_size=1)

    output = model.predict(train_x[:1])
    plt.imshow(output[0])
    plt.show()
