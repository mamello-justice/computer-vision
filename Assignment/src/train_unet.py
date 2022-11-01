import os
from os import path

import numpy as np

from keras import callbacks
from segmentation_models import Unet

import common

BACKBONE = 'vgg16'

data_dir = path.abspath(path.join(path.dirname(__file__), '..', 'data'))

default_cp_path = path.join(data_dir, 'cp.ckpt')
default_model_path = path.join(data_dir, 'model.h5')


def setup_args():
    parser = common.default_parser(
        description=f'Deep Learning (UNet')

    parser.add_argument('--cp-path',
                        default=default_cp_path,
                        help='path for checkpoint file')

    parser.add_argument('--model-path',
                        default=default_model_path,
                        help='path for model file')

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = setup_args()

    use_cpu = args['cpu']
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cp_path = args['cp_path']
    input_dir = args['input_dir']
    input_size = args['input_size']
    model_path = args['model_path']

    height, width = input_size

    raw_images, raw_masks = common.load_images(input_dir, height, width)

    masks = np.expand_dims(raw_masks, axis=3)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) =\
        common.split_data(raw_images, masks)

    checkpoint_cb = callbacks.ModelCheckpoint(filepath=cp_path,
                                              save_weights_only=True,
                                              verbose=1)

    model = Unet(BACKBONE,
                 encoder_weights='imagenet',
                 input_shape=(*input_size, 3))
    model.compile('Adam', 'binary_crossentropy')

    model.fit(train_x, train_y,
              batch_size=2,
              epochs=4,
              validation_data=(val_x, val_y),
              callbacks=[checkpoint_cb])

    model.save(model_path)
