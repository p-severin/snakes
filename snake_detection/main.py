import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

from utils.image import ImageHelper
from snake_detection import logger
from snake_detection.dataset import Dataset
from snake_detection.model import Model


def compare_rotations():
    python1_path = os.path.join(Dataset.EXTRACTED_DATASET_DIRECTORY, '1_extracted.png')
    python0 = ImageHelper.open_image(python1_path)
    print(python0.shape)
    tile0 = ImageHelper.get_tile_from_image(python0, 80, 100, (100, 100))
    rotated1 = rotate(tile0, 45, axes=(1, 0), reshape=True)
    rotated2 = rotate(tile0, 45, axes=(1, 0), reshape=False)

    _, axes = plt.subplots(1, 3)
    axes[0].imshow(tile0)
    axes[1].imshow(rotated1)
    axes[2].imshow(rotated2)
    plt.show()


def run(training=True):
    # All images: 22
    num_training_images = 1
    num_test_images = 1

    tile_size = (64, 64)
    stride = 8
    threshold = 0.5
    model_path = '../repository/models/snake_piotr2.model'

    logger.info("Loading dataset")
    dataset = Dataset(num_training_images, num_test_images)
    dataset.load()
    dataset.prepare_xy(tile_size, stride, threshold)

    logger.debug("Size of original: %s extracted: %s ",
                 len(dataset.original), len(dataset.extracted))
    logger.debug("Number of ORIGINAL train tiles: %s test tiles: %s, size of tile: %s",
                 len(dataset.X_train), len(dataset.X_test), dataset.X_train[0].shape)
    logger.debug("Number of EXTRACTED train tiles: %s test tiles: %s",
                 len(dataset.Y_train), len(dataset.Y_test))

    snake_tiles = {
        'train': np.sum(dataset.Y_train) / dataset.Y_train.size,
        'test': np.sum(dataset.Y_test) / dataset.Y_test.size,
    }

    for subset in ['train', 'test']:
        c = snake_tiles[subset]
        proportions = (max(1, c / (1 - c)), max(1, (1 - c) / c))
        logger.debug("Tiles in %s set: %.2f%%", subset, c * 100)
        logger.debug("Proportion of snake to not-snake in %s set: %.2f to %.2f",
                     subset, *proportions)

    model = Model(dataset, model_path)
    if training:
        logger.info("Creating model")
        model.create(tile_size)
        logger.info("Training model")
        model.train(epochs=1, verbose=1)
        logger.info("Saving model")
        model.save()
    else:
        logger.info("Loading model")
        model.load()
    # model.evaluate()
    results = model.predict(verbose=1)

    # pass tiles of one image to this
    heatmap, result_image = dataset.get_heatmap(dataset.original[-1], results, 0.90)



if __name__ == "__main__":
    # compare_rotations()
    run(training=False)
