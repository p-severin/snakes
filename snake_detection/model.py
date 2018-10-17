import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import (
    Dense, Dropout, MaxPool2D, Flatten, Conv2D, BatchNormalization
)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from snake_detection import logger


class Model:
    def __init__(self, dataset, model_path):
        self.dataset = dataset
        self.model = Sequential()
        self.model_path = model_path

    def create(self, tile_size: (int, int)):
        input_shape = (*tile_size, 3)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu',
                         padding='same', input_shape=input_shape))
        # model.add(Conv2D(filters=32, kernel_size=3,
        #                  padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D())
        model.add(Conv2D(filters=64, kernel_size=3,
                         padding='same', activation='relu'))
        # model.add(Conv2D(filters=64, kernel_size=3,
        #                  padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        model.summary()

        self.model = model

    def train(self, epochs, verbose=2):
        train_datagen = self._get_scaling_generator(self.dataset.X_train,
                                                    self.dataset.Y_train)
        val_datagen = self._get_scaling_generator(self.dataset.X_test,
                                                  self.dataset.Y_test)

        train_history = self.model.fit_generator(train_datagen,
                                                 epochs=epochs,
                                                 validation_data=val_datagen,
                                                 verbose=verbose)

        fig, axes = plt.subplots(1, 2, sharex='all')
        axes[0].plot(train_history.history['loss'])
        axes[0].plot(train_history.history['val_loss'])
        axes[0].set_ylabel('loss')
        axes[1].plot(train_history.history['acc'])
        axes[1].plot(train_history.history['val_acc'])
        axes[1].set_ylabel('accuracy')
        axes[0].legend(['train loss', 'validation loss'], loc='best')
        axes[1].legend(['train accuracy', 'validation accuracy'], loc='best')
        fig.text(0.5, 0.02, "Number of epochs", horizontalalignment='center')

        plt.show()

    def evaluate(self):
        test_datagen = self._get_scaling_generator(self.dataset.X_test,
                                                   self.dataset.Y_test)
        outputs = self.model.evaluate_generator(test_datagen)
        logger.info("Results: Loss: %.3f, Accuracy: %.3f", *outputs)

    def predict(self, verbose=1):
        test_datagen = self._get_scaling_generator(self.dataset.X_test, shuffle=False)
        results = self.model.predict_generator(test_datagen, verbose=verbose)
        return results

    def save(self):
        model_path = self.model_path
        self.model.save(model_path)

    def load(self):
        model_path = self.model_path
        self.model = load_model(model_path)

    @staticmethod
    def _get_scaling_generator(x, y=None, shuffle=True):
        datagen = ImageDataGenerator(rescale=1. / 255)
        return datagen.flow(x, y, shuffle=shuffle)
