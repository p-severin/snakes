import logging
import os
from itertools import product
from typing import List

import numpy as np
from PIL import Image, ImageOps

from utils.image import ImageHelper

logger = logging.getLogger("pythons")


class Dataset:
    _PROJECT_DIRECTORY = '/home/snake_detection/snake-distinction'
    ORIGINAL_DATASET_DIRECTORY = \
        os.path.join(_PROJECT_DIRECTORY, 'repository/indian_rock_python/original')
    EXTRACTED_DATASET_DIRECTORY = \
        os.path.join(_PROJECT_DIRECTORY, 'repository/indian_rock_python/extracted')
    NUM_DATASET_IMAGES = 22
    INPUT_SIZE = (512, 512, 3)

    def __init__(self, n_train_images, n_test_images):
        assert n_train_images + n_test_images <= self.NUM_DATASET_IMAGES
        self.extracted = []
        self.original = []
        self.X = {'train': [], 'test': []}
        self.y = {'train': [], 'test': []}
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.TILE_SIZE = (0, 0)
        self.STRIDE = 0
        self.NUM_TRAIN_IMAGES = n_train_images
        self.NUM_TEST_IMAGES = n_test_images

    def get_image_paths(self, dataset='original'):

        if dataset not in ['original', 'extracted']:
            raise ValueError("Invalid dataset, try 'original' | 'extracted' ")

        if dataset == 'original':
            paths = os.listdir(self.ORIGINAL_DATASET_DIRECTORY)
            paths.sort(key=lambda x: int(x.split('_')[0]))
            paths = [os.path.join(self.ORIGINAL_DATASET_DIRECTORY, path) for path in paths]
        else:
            paths = os.listdir(self.EXTRACTED_DATASET_DIRECTORY)
            paths.sort(key=lambda x: int(x.split('_')[0]))
            paths = [os.path.join(self.EXTRACTED_DATASET_DIRECTORY, path) for path in paths]

        assert len(paths) == self.NUM_DATASET_IMAGES
        return paths

    def load(self):
        original_paths = self.get_image_paths('original')
        extracted_paths = self.get_image_paths('extracted')

        self.original = [np.array(Image.open(path)) for path in original_paths]
        self.extracted = [np.array(Image.open(path)) for path in extracted_paths]

    @staticmethod
    def convert_dataset_to_tiles(dataset: List[np.ndarray], tile_size: (int, int), stride: int):
        tiles = []
        for i, image in enumerate(dataset):
            tiles_one_image = ImageHelper.get_tiles_from_image(image, tile_size, stride)
            tiles.extend(tiles_one_image)
        return np.array(tiles, np.uint8)

    @staticmethod
    def convert_tiles_to_class_vector(tiles, threshold: float):

        class_vector = np.array(
            [ImageHelper.classify_extracted_tile(tile, threshold) for tile in tiles],
            dtype=np.uint8)
        return class_vector

    def prepare_xy(self, tile_size, stride, threshold):

        self._resize_images()
        assert isinstance(tile_size, tuple)
        self.TILE_SIZE = tile_size
        self.STRIDE = stride

        logger.debug("converting original images to input data")
        self.X_train = self.convert_dataset_to_tiles(
            self.original[:self.NUM_TRAIN_IMAGES], tile_size, stride)
        self.X_test = self.convert_dataset_to_tiles(
            self.original[-self.NUM_TEST_IMAGES:], tile_size, stride)

        logger.debug("converting extracted images to tiles")
        y_train_tiles = self.convert_dataset_to_tiles(
            self.extracted[:self.NUM_TRAIN_IMAGES], tile_size, stride)
        y_test_tiles = self.convert_dataset_to_tiles(
            self.extracted[-self.NUM_TEST_IMAGES:], tile_size, stride)

        logger.debug("converting extracted tiles to ground truth")
        self.Y_train = self.convert_tiles_to_class_vector(y_train_tiles, threshold)
        self.Y_test = self.convert_tiles_to_class_vector(y_test_tiles, threshold)

    def _resize_images(self):
        self.original = np.array([ImageHelper.resize_image(image, self.INPUT_SIZE)
                                  for image in self.original], np.uint8)
        self.extracted = np.array([ImageHelper.resize_image(image, self.INPUT_SIZE)
                                   for image in self.extracted], np.uint8)

    def get_heatmap(self, image_x, image_y, heatmap_opacity=1):

        img_size = self.INPUT_SIZE[:-1]
        heatmap = np.zeros(img_size, dtype=np.float32)
        overlap_counts = np.zeros(img_size, dtype=np.uint8)

        tiles_coordinates = product(
            np.arange(0, self.INPUT_SIZE[0] - self.TILE_SIZE[0] + 1, self.STRIDE),
            np.arange(0, self.INPUT_SIZE[1] - self.TILE_SIZE[1] + 1, self.STRIDE))

        for i, coord in enumerate(tiles_coordinates):
            y, x = coord
            heatmap[y:y + self.TILE_SIZE[0], x:x + self.TILE_SIZE[1]] += image_y[i]
            overlap_counts[y:y + self.TILE_SIZE[0], x:x + self.TILE_SIZE[1]] += 1

        overlap_counts[overlap_counts == 0] = 1
        heatmap /= overlap_counts

        heatmap_mask = heatmap * 255
        image = Image.fromarray(image_x)

        heatmap_image = Image.fromarray(heatmap * 255).convert('L')
        heatmap_image = ImageOps.colorize(heatmap_image, 'black', 'red')

        heatmap_mask[heatmap < 0.5] = 0
        heatmap_mask = (heatmap_mask * heatmap_opacity).astype(np.uint8)
        heatmap_image.putalpha(Image.fromarray(heatmap_mask))
        image.putalpha(Image.new('L', image.size, int((1 - heatmap_opacity) * 255)))

        result_image = np.array(Image.alpha_composite(heatmap_image, image))
        return heatmap, result_image

    def save_images(self, directory):
        pass
