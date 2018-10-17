import logging
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, MaxPool2D, Flatten, Conv2D, BatchNormalization

import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras.utils import plot_model

FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATEFMT = '%H:%M:%S'

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFMT))
logger.addHandler(ch)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



class NetworkModel:
    def __init__(self, main_directory, dataset, input_shape):
        self.MAIN_DIRECTORY = main_directory
        self.dataset = dataset
        self.input_shape = input_shape

    def create_model(self):
        logger.info('Creating keras model.')

        input_shape = self.input_shape
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D())
        model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # SVG(model_to_dot(model).create(prog='dot', format='svg'))
        model.summary()
        self.model = model

    def train_model(self, epochs=5):
        # class_weights = class_weight.compute_class_weight(class_weight='balanced',
        #                                                   classes=np.unique(self.dataset.y['train'], axis=0),
        #                                                   y=self.dataset.y['train'])
        # logger.info('CLASS_WEIGHTS: {}'.format(class_weights))

        file_path = self.MAIN_DIRECTORY + '/repo/models/snake_model_{epoch:02d}_{val_categorical_accuracy:.2f}.model'
        # file_path = self.MAIN_DIRECTORY + '/repo/models/snake_model.model'
        checkpoint = ModelCheckpoint(filepath=file_path,
                                     monitor='val_categorical_accuracy',
                                     save_best_only=True,
                                     save_weights_only=False,
                                     verbose=1)

        train_history = self.model.fit(x=self.dataset.X['train'],
                                       y=self.dataset.y['train'],
                                       epochs=epochs,
                                       validation_data=(self.dataset.X['test'], self.dataset.y['test']),
                                       batch_size=32,
                                       verbose=1,
                                       callbacks=[checkpoint],
                                       class_weight='auto'
                                       )

        logger.info(train_history.history)

        plt.plot(train_history.history['categorical_accuracy'])
        plt.plot(train_history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def save_model(self):
        fn = self.MAIN_DIRECTORY + '/repo/models/snake_model.model'
        logger.info('Model saved to file: {}'.format(fn))
        self.model.save(fn)

    def load_model(self):
        fn = self.MAIN_DIRECTORY + '/repo/models/snake_model_03_0.78.model'
        logger.info('Model loaded from file: {}'.format(fn))
        self.model = load_model(fn)
