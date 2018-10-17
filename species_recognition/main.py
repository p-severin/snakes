import glob

import matplotlib
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize

from species_recognition.data import Data
from species_recognition.neural_network import NetworkModel
from species_recognition.predictor import Predictor
from utils.image import ImageHelper
from utils.plotter import Plotter




FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATEFMT = '%H:%M:%S'

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFMT))
logger.addHandler(ch)

MAIN_DIRECTORY = os.getcwd()
SNAKES_SPECIES = ['king_cobra', 'indian_rock_python']
STRIDE = 16
TILE_SIZE = (64, 64, 3)
IMAGE_SIZE = (512, 512, 3)
TRAIN_TO_TEST_SPLIT_RATIO = 0.8
TILES_TAKEN_FROM_IMAGE_RATIO = 0.6
HOW_MANY_IMAGES_USED = -1
HOW_MANY_ROTATED_IMAGES = 5
ImageHelper.PINK_THRESHOLD = 0.15

prepare_data = False
save_data = False
load_data = False
load_model = True
train_model = False
predict = True
plot = True
save_results = False

data = Data(main_directory=MAIN_DIRECTORY,
            species=SNAKES_SPECIES,
            stride=STRIDE,
            tile_size=TILE_SIZE,
            image_size=IMAGE_SIZE,
            train_to_test_split=TRAIN_TO_TEST_SPLIT_RATIO,
            tiles_taken_from_image_ratio=TILES_TAKEN_FROM_IMAGE_RATIO,
            how_many_images_used=HOW_MANY_IMAGES_USED,
            how_many_rotated_images=HOW_MANY_ROTATED_IMAGES)
model = NetworkModel(main_directory=MAIN_DIRECTORY,
                     dataset=data,
                     input_shape=TILE_SIZE)
predictor = Predictor(main_directory=MAIN_DIRECTORY,
                      model=model,
                      data=data,
                      tile_size=TILE_SIZE,
                      stride=STRIDE)
plotter = Plotter(main_directory=MAIN_DIRECTORY,
                  dataset=data,
                  predictor=predictor)

test_images = [imresize(np.array(Image.open(path)), size=(512, 512, 3)) for path in glob.glob('/home/pseweryn/Projects/Snakes/repo/test_images/*')]


def run():
    if prepare_data:
        data.prepare_file_paths_for_images()
        data.prepare_dataset()
        logger.debug('X train shape: {}'.format(data.X['train'].shape))
        logger.debug('y train shape: {}'.format(data.y['train'].shape))
        logger.debug('X test shape: {}'.format(data.X['test'].shape))
        logger.debug('y test shape: {}'.format(data.y['test'].shape))
        unique_records, counts = np.unique(data.y['train'], axis=0, return_counts=True)
        print('Unique records in dataset: {}, Count of unique records: {}'.format(unique_records, counts))
    if save_data:
        data.save_npz()
    if load_data:
        data.load_npz()
    if load_model:
        model.load_model()
    else:
        model.create_model()
    if train_model:
        model.train_model(epochs=10)
    if predict:
        predictor.predict_all_images()
        # results = predictor.predict_unknown_images(test_images[0])
    if plot:
        # plotter.plot_is_snake_or_not()
        # plotter.plot_test_images(results)
        plotter.plot_snake_detection_points(save=True)
    if save_results:
        data.save_npz_results()

def recreate_image_with_classification():
    snake_type = 'indian_rock_python'
    subset = 'test'
    snake_number = 3
    final_image = data.recreate_output_image(data.X_image[snake_type][subset][snake_number],
                                             data.y_image[snake_type][subset][snake_number], snake_type)
    matplotlib.rcParams.update({'font.size': 9})
    plt.subplot(121)
    plt.imshow(data.X_image[snake_type][subset][snake_number])
    plt.axis('off')
    plt.title('Original image')
    plt.subplot(122)
    plt.imshow(final_image)
    plt.axis('off')
    plt.title('Snake position prediction with tiles')
    # plt.show()
    plt.savefig(
        '{}{}.{}'.format(MAIN_DIRECTORY + '/repo/visuals/', 'ground_truth', 'png'),
        bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run()
    # recreate_image_with_classification()
