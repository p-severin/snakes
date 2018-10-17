import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from utils.image import ImageHelper
import numpy as np
import logging

FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATEFMT = '%H:%M:%S'

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFMT))
logger.addHandler(ch)


class Data:
    def __init__(self, main_directory, species, stride, tile_size, image_size, train_to_test_split,
                 tiles_taken_from_image_ratio, how_many_images_used=-1, how_many_rotated_images=3):
        self.HOW_MANY_ROTATED_IMAGES = how_many_rotated_images
        self.MAIN_DIRECTORY = main_directory
        self.species = species
        self.STRIDE = stride
        self.TILE_SIZE = tile_size
        self.SIZE = image_size
        self.TRAIN_TO_TEST_SPLIT = train_to_test_split
        self.TILES_TAKEN_FROM_IMAGE_RATIO = tiles_taken_from_image_ratio
        self.HOW_MANY_IMAGES_USED = how_many_images_used
        self.X = {'train': [], 'test': []}
        self.y_tiles = {'train': [], 'test': []}
        self.y = {'train': [], 'test': []}
        self.X_image = {key: {} for key in species}
        self.y_image = {key: {} for key in species}
        self.predictions = {'no_snake': [], 'king_cobra': [], 'indian_rock_python': [], 'original': []}
        self.original_files_paths = {key: {} for key in species}
        self.free_form_files_paths = {key: {} for key in species}
        self.original_files_directory = {key: self.MAIN_DIRECTORY + '/repo/' + key + '/original/' for key in species}
        self.free_form_files_directory = {key: self.MAIN_DIRECTORY + '/repo/' + key + '/extracted/' for key in species}
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    def __get_paths_for_images_in_both_folders(self, snake_type):
        original_files = os.listdir(self.original_files_directory[snake_type])
        free_form_files = os.listdir(self.free_form_files_directory[snake_type])
        paths_to_original_images = []
        paths_to_free_form_images = []
        for free_form_file in free_form_files:
            if any(free_form_file[:-4] in string for string in original_files):
                paths_to_original_images.append(self.original_files_directory[snake_type] + free_form_file)
                paths_to_free_form_images.append(self.free_form_files_directory[snake_type] + free_form_file)

        if self.HOW_MANY_IMAGES_USED == -1:
            return paths_to_original_images, paths_to_free_form_images
        else:
            return paths_to_original_images[:self.HOW_MANY_IMAGES_USED], \
                   paths_to_free_form_images[:self.HOW_MANY_IMAGES_USED]

    def __set_paths(self, paths_to_original_images, paths_to_free_form_images, snake_type):
        paths_to_original_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        paths_to_free_form_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.original_files_paths[snake_type]['train'] = paths_to_original_images[
                                                         :int(self.TRAIN_TO_TEST_SPLIT * len(paths_to_original_images))]
        self.original_files_paths[snake_type]['test'] = paths_to_original_images[
                                                        int(self.TRAIN_TO_TEST_SPLIT * len(paths_to_original_images)):]
        self.free_form_files_paths[snake_type]['train'] = paths_to_free_form_images[
                                                          :int(self.TRAIN_TO_TEST_SPLIT * len(
                                                              paths_to_free_form_images))]
        self.free_form_files_paths[snake_type]['test'] = paths_to_free_form_images[
                                                         int(self.TRAIN_TO_TEST_SPLIT * len(
                                                             paths_to_free_form_images)):]
        logger.info(
            'Train size: {} for snake: {}'.format(len(self.original_files_paths[snake_type]['train']), snake_type))
        logger.info(
            'Test size: {} for snake: {}'.format(len(self.original_files_paths[snake_type]['test']), snake_type))

    def __open_images(self, subset, snake_type):
        logger.info('Opening images from paths for snake: {} and subset: {}.'.format(snake_type, subset))
        self.X_image[snake_type][subset] = [ImageHelper.open_image(path) for path in
                                            self.original_files_paths[snake_type][subset]]
        self.y_image[snake_type][subset] = [ImageHelper.open_image(path) for path in
                                            self.free_form_files_paths[snake_type][subset]]

    def __resize_images(self, subset, snake_type):
        logger.info('Resizing images for snake: {} and subset: {}.'.format(snake_type, subset))
        resized_X_images = [ImageHelper.resize_image(image, self.SIZE) for image in self.X_image[snake_type][subset]]
        resized_y_images = [ImageHelper.resize_image(image, self.SIZE) for image in self.y_image[snake_type][subset]]
        self.X_image[snake_type][subset] = resized_X_images
        self.y_image[snake_type][subset] = resized_y_images

    def __create_modified_images(self, subset, snake_type, number_of_copies):
        logger.info(
            'Creating modified images (rotation, flipping, color changing, etc.) for snake: {} and subset: {}'.format(
                snake_type, subset))
        total_number_of_pictures = len(self.X_image[snake_type][subset])
        for i in range(total_number_of_pictures):
            image_X = self.X_image[snake_type][subset][i]
            image_y = self.y_image[snake_type][subset][i]
            for j in range(number_of_copies):
                new_image_X = image_X
                new_image_y = image_y

                color_factor = np.random.uniform(0.7, 1.3)
                new_image_X = ImageHelper.enhance_color(new_image_X, color_factor)

                brightness_factor = np.random.uniform(0.7, 1.3)
                new_image_X = ImageHelper.enhance_brightness(new_image_X, brightness_factor)

                sharpness_factor = np.random.uniform(0.7, 1.3)
                new_image_X = ImageHelper.enhance_sharpness(new_image_X, sharpness_factor)

                contrast_factor = np.random.uniform(0.7, 1.3)
                new_image_X = ImageHelper.enhance_contrast(new_image_X, contrast_factor)

                should_flip = np.random.choice([True, False])
                if should_flip:
                    axis = np.random.choice([0, 1])
                    new_image_X = ImageHelper.flip_image(new_image_X, axis)
                    new_image_y = ImageHelper.flip_image(new_image_y, axis)

                angle = np.random.randint(0, 360)
                new_image_X = ImageHelper.rotate_image(new_image_X, angle)
                new_image_y = ImageHelper.rotate_image(new_image_y, angle)

                self.X_image[snake_type][subset].append(new_image_X)
                self.y_image[snake_type][subset].append(new_image_y)

    def __convert_images_to_numpy_arrays(self, subset, snake_type):
        logger.info('Converting data to numpy arrays.')
        X_images = np.array(
            [ImageHelper.convert_image_to_numpy_array(image) for image in self.X_image[snake_type][subset]])
        y_images = np.array(
            [ImageHelper.convert_image_to_numpy_array(image) for image in self.y_image[snake_type][subset]])
        self.X_image[snake_type][subset] = X_images
        self.y_image[snake_type][subset] = y_images

    def __convert_numpy_arrays_to_images(self, subset, snake_type):
        logger.info('Converting numpy arrays to PIL images.')
        X_images = [ImageHelper.convert_numpy_array_to_image(image) for image in self.X_image[snake_type][subset]]
        y_images = [ImageHelper.convert_numpy_array_to_image(image) for image in self.y_image[snake_type][subset]]
        self.X_image[snake_type][subset] = X_images
        self.y_image[snake_type][subset] = y_images

    def __set_Xy_dataset(self, subset, snake_type):
        length = len(self.X_image[snake_type][subset])
        for i in range(length):
            image = self.X_image[snake_type][subset][i]
            y_image = self.y_image[snake_type][subset][i]

            tiles_X = ImageHelper.get_tiles_from_image(image=image,
                                                       tile_size=self.TILE_SIZE,
                                                       stride=self.STRIDE)
            tiles_y = ImageHelper.get_tiles_from_image(image=y_image,
                                                       tile_size=self.TILE_SIZE,
                                                       stride=self.STRIDE)

            how_many_tiles = len(tiles_X)
            indexes = np.random.choice(range(how_many_tiles),
                                       size=int(self.TILES_TAKEN_FROM_IMAGE_RATIO * how_many_tiles), replace=False)
            tiles_X = [tiles_X[i] for i in indexes]
            tiles_y = [tiles_y[i] for i in indexes]
            snake_responses = [ImageHelper.get_snake_number(tile, snake_type) for tile in tiles_y]

            self.X[subset].extend(tiles_X)
            self.y_tiles[subset].extend(tiles_y)
            self.y[subset].extend(snake_responses)

    def __one_hot_encode_y(self, subset):
        integer_encoded = self.label_encoder.fit_transform(self.y[subset])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded = self.one_hot_encoder.fit_transform(integer_encoded)
        self.y[subset] = one_hot_encoded

    def __data_to_numpy_array(self, subset):
        self.X[subset] = np.array(self.X[subset], dtype=np.uint8)
        self.y[subset] = np.array(self.y[subset], dtype=np.uint8)

    def __normalize_data(self, subset):
        self.X[subset] = np.divide(self.X[subset], 255)
        self.y_tiles[subset] = np.divide(self.y_tiles[subset], 255)

    def prepare_file_paths_for_images(self):
        logger.info('Reading file paths for images.')
        for snake_type in self.species:
            paths_to_original_images, paths_to_free_form_images = self.__get_paths_for_images_in_both_folders(
                snake_type)
            self.__set_paths(paths_to_original_images, paths_to_free_form_images, snake_type)

    def prepare_dataset(self):
        logger.info('PREPARING DATASET.')
        for subset in ['train', 'test']:
            for snake_type in self.species:
                self.__open_images(subset, snake_type)
                self.__resize_images(subset, snake_type)
                self.__convert_numpy_arrays_to_images(subset, snake_type)
                self.__create_modified_images(subset, snake_type, number_of_copies=self.HOW_MANY_ROTATED_IMAGES)
                self.__convert_images_to_numpy_arrays(subset, snake_type)
                self.__set_Xy_dataset(subset, snake_type)
            self.__one_hot_encode_y(subset)
            self.__data_to_numpy_array(subset)

    def save_npz(self):
        fn = './repo/datasets/snakes.npz'
        logger.info('Saving dataset to: {}'.format(fn))
        np.savez_compressed(file=fn,
                            X_train=self.X['train'],
                            y_train=self.y['train'],
                            X_test=self.X['test'],
                            y_test=self.y['test'],
                            )

    def save_npz_results(self):
        fn = './repo/datasets/snakes_results.npz'
        logger.info('Saving dataset to: {}'.format(fn))
        np.savez_compressed(file=fn,
                            no_snake=self.predictions['no_snake'],
                            king_cobra=self.predictions['king_cobra'],
                            indian_rock_python=self.predictions['indian_rock_python'],
                            original=self.predictions['original']
                            )

    def load_npz(self):
        fn = './repo/datasets/snakes.npz'
        logger.info('Loading dataset from: {}'.format(fn))
        loaded_data = np.load(fn)
        self.X['train'] = loaded_data['X_train']
        self.y['train'] = loaded_data['y_train']
        self.X['test'] = loaded_data['X_test']
        self.y['test'] = loaded_data['y_test']

    def recreate_output_image(self, image, y_image, snake_type):
        shape = image.shape[:-1]
        predicted_image = np.zeros(shape)
        image_overlap_matrix = np.zeros(shape)
        tiles = ImageHelper.get_tiles_from_image(image=image,
                                                 tile_size=self.TILE_SIZE,
                                                 stride=self.STRIDE)
        tiles_y = ImageHelper.get_tiles_from_image(image=y_image,
                                                   tile_size=self.TILE_SIZE,
                                                   stride=self.STRIDE)
        snake_responses = [ImageHelper.get_snake_number(tile, snake_type) for tile in tiles_y]
        how_many_tiles_in_y = int(np.floor((shape[0] - self.TILE_SIZE[0]) / self.STRIDE) + 1)
        how_many_tiles_in_x = int(np.floor((shape[1] - self.TILE_SIZE[1]) / self.STRIDE) + 1)

        for tile in tiles:
            assert tile.shape[0] == self.TILE_SIZE[0] and tile.shape[1] == self.TILE_SIZE[1]
        assert len(tiles) == how_many_tiles_in_y * how_many_tiles_in_x

        for y in range(how_many_tiles_in_y):
            for x in range(how_many_tiles_in_x):
                snake_index_chosen = snake_responses[y * how_many_tiles_in_x + x]
                x_pos = x * self.STRIDE
                y_pos = y * self.STRIDE
                predicted_image[y_pos: y_pos + self.TILE_SIZE[0],
                x_pos: x_pos + self.TILE_SIZE[1]] += snake_index_chosen
                image_overlap_matrix[y_pos: y_pos + self.TILE_SIZE[0], x_pos: x_pos + self.TILE_SIZE[1]] += 1

        image_overlap_matrix[image_overlap_matrix == 0] = 1
        return predicted_image / image_overlap_matrix
