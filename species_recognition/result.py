import numpy as np


class Result:

    @staticmethod
    def combine_predictions(image_with_snake_position, image_with_snake_type):
        return np.multiply(image_with_snake_position, image_with_snake_type)