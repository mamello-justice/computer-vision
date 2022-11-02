import numpy as np


def split_data(x, y):
    assert len(x) == len(y), "size of x and y must be equal"

    N = len(x)
    indices = np.random.permutation(N)

    train_end = int(0.7*N) + 1
    train_indices = indices[:train_end]

    validation_end = train_end + int(0.15*N)
    validation_indices = indices[train_end:validation_end]

    test_indices = indices[validation_end:]

    return (x[train_indices], y[train_indices]),\
        (x[validation_indices], y[validation_indices]),\
        (x[test_indices], y[test_indices])


def preprocess_data(x, y, features):
    x_new = np.empty((*x.shape[:-1], 0))
    y_new = y

    if len(y_new.shape) == 3:
        y_new = np.expand_dims(y_new, axis=3)

    if 'rgb' in features:
        x_new = np.append(x_new, x, axis=3)

    if 'DoG' in features:
        import cv2

        sigma = 1
        K = 3
        kernel_shape = (49, 49)

        temp = []

        for i in range(len(x_new)):
            G1 = cv2.GaussianBlur(x_new[i], ksize=kernel_shape, sigmaX=sigma)
            G2 = cv2.GaussianBlur(
                x_new[i], ksize=kernel_shape, sigmaX=K**2 * sigma)
            temp.append(G2 - G1)

        x_new = np.append(x_new, temp, axis=3)

    return x_new, y_new
