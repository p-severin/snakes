import logging
import numpy as np
import matplotlib.pyplot as plt

from utils.image import ImageHelper

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, main_directory, model, data, tile_size, stride):
        self.MAIN_DIRECTORY = main_directory
        self.model = model
        self.data = data
        self.tile_size = tile_size
        self.stride = stride
        self.results = []

    def predict(self, image):
        image = image.reshape((1, 64, 64, 3))
        predicted_snake = self.model.model.predict(image)
        return predicted_snake

    def predict_image(self, image):
        shape = image.shape[:-1]
        predicted_no_snake = np.zeros(shape)
        predicted_cobra = np.zeros(shape)
        predicted_python = np.zeros(shape)
        image_overlap_matrix = np.zeros(shape)
        tiles = ImageHelper.get_tiles_from_image(image, tile_size=self.tile_size, stride=self.stride)

        how_many_tiles_in_y = int(np.floor((shape[0] - self.tile_size[0]) / self.stride) + 1)
        how_many_tiles_in_x = int(np.floor((shape[1] - self.tile_size[1]) / self.stride) + 1)

        assert len(tiles) == how_many_tiles_in_y * how_many_tiles_in_x
        for tile in tiles:
            assert tile.shape[0] == 64 and tile.shape[1] == 64

        for y in range(how_many_tiles_in_y):
            for x in range(how_many_tiles_in_x):
                tile = tiles[y * how_many_tiles_in_x + x]
                predicted_tile = self.predict(tile)
                x_pos = x * self.stride
                y_pos = y * self.stride
                predicted_no_snake[y_pos: y_pos + self.tile_size[0], x_pos: x_pos + self.tile_size[1]] += predicted_tile[0][0]
                predicted_cobra[y_pos: y_pos + self.tile_size[0], x_pos: x_pos + self.tile_size[1]] += predicted_tile[0][1]
                predicted_python[y_pos: y_pos + self.tile_size[0], x_pos: x_pos + self.tile_size[1]] += predicted_tile[0][2]
                image_overlap_matrix[y_pos: y_pos + self.tile_size[0], x_pos: x_pos + self.tile_size[1]] += 1

        image_overlap_matrix[image_overlap_matrix == 0] = 1
        predicted_no_snake /= image_overlap_matrix
        predicted_cobra /= image_overlap_matrix
        predicted_python /= image_overlap_matrix
        return predicted_no_snake, predicted_cobra, predicted_python, image

    def predict_all_images(self):
        no_snake = []
        king_cobra = []
        indian_rock_python = []
        original = []
        for snake_type in self.data.species:
            for image in self.data.X_image[snake_type]['test']:
                predictions = self.predict_image(image)
                no_snake.append(predictions[0])
                king_cobra.append(predictions[1])
                indian_rock_python.append(predictions[2])
                original.append(predictions[3])
        self.data.predictions['no_snake'] = np.array(no_snake, np.float32)
        self.data.predictions['king_cobra'] = np.array(king_cobra, np.float32)
        self.data.predictions['indian_rock_python'] = np.array(indian_rock_python, np.float32)
        self.data.predictions['original'] = np.array(original, np.uint8)

    def predict_unknown_images(self, image):
        no_snake = []
        king_cobra = []
        indian_rock_python = []
        original = []
        # for image in images:
        predictions = self.predict_image(image)
        no_snake.append(predictions[0])
        king_cobra.append(predictions[1])
        indian_rock_python.append(predictions[2])
        original.append(predictions[3])
        return (no_snake, king_cobra, indian_rock_python, original)




