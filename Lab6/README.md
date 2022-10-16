# Computer Vision

## Lab 6

### Requirements

- [Jupyter](https://jupyter.org)
- [Puzzle corners](./docs/puzzle_corners_1024x768__v2.zip)

### Libraries used

- [NumPy](https://numpy.org/) - Various array/matrix operations
- [Matplotlib](https://matplotlib.org/) - Displaying images and plotting charts
- [OpenCV](https://opencv.org/) - ImageIO, color conversion, threshold, contour operations and drawing contours
- [SciPy](https://scipy.org/) - Using KDTree for getting contour points closest to corners
- [SciKit Image](https://scikit-image.org/) - Image color and dtype conversions
- [SciKit Learn](https://scikit-learn.org/) - Metrics like MSE and calculating the nearest neighbors
- [NetworkX](https://networkx.org/) - For creating graphs

### Getting started

1. Extract puzzle images, masks and corner data to [assets](./assets) dir
2. Install dependencies (ideally in a virtual environment [virtualenv](https://github.com/pypa/virtualenv) or [pyenv](https://github.com/pyenv/pyenv))
3. Run jupyter notebook/lab
