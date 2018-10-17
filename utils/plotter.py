import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12})


class Plotter:
    def __init__(self, main_directory, dataset, predictor):
        self.MAIN_DIRECTORY = main_directory
        self.dataset = dataset
        self.predictor = predictor

    def plot_test_images(self, results, save=True):

        figure, axes = plt.subplots(1, 4, figsize=(15, 15))

        axes[0].imshow(results[0][0])
        axes[0].set_title('no snake')
        axes[0].axis('off')

        axes[1].imshow(results[1][0], vmin=0, vmax=1)
        axes[1].set_title('cobra')
        axes[1].axis('off')

        axes[2].imshow(results[2][0], vmin=0, vmax=1)
        axes[2].set_title('python')
        axes[2].axis('off')

        axes[3].imshow(results[3][0], vmin=0, vmax=1)
        axes[3].set_title('original')
        axes[3].axis('off')

        plt.tight_layout()
        if save:
            plt.savefig(
                '{}{}.{}'.format(self.MAIN_DIRECTORY + '/repo/visuals/predictions/', 'test_image', 'png'),
                bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_snake_detection_points(self, save=False):
        for snake_type in self.dataset.species:
            for i in range(len(self.dataset.X_image[snake_type]['test'])):
                figure, axes = plt.subplots(1, 4, figsize=(15, 15))
                image = self.dataset.X_image[snake_type]['test'][i]
                predicted_images = self.predictor.predict_image(image)

                axes[0].imshow(image)
                axes[0].set_title(snake_type)
                axes[0].axis('off')

                axes[1].imshow(predicted_images[0], vmin=0, vmax=1)
                axes[1].set_title('no snake')
                axes[1].axis('off')

                axes[2].imshow(predicted_images[1], vmin=0, vmax=1)
                axes[2].set_title('cobra')
                axes[2].axis('off')

                axes[3].imshow(predicted_images[2], vmin=0, vmax=1)
                axes[3].set_title('python')
                axes[3].axis('off')

                plt.tight_layout()
                if save:
                    plt.savefig(
                        '{}{}.{}'.format(self.MAIN_DIRECTORY + '/repo/visuals/predictions/', snake_type + '_' + str(i), 'png'),
                        bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

    def plot_is_snake_or_not(self, save=True):
        indexes = range(len(self.dataset.y['test'][:100]))
        matplotlib.rcParams.update({'font.size': 9})
        for i in indexes:
            plt.subplot(121)
            plt.imshow(self.dataset.X['test'][i])
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')
            plt.title('Original tile')

            plt.subplot(122)
            plt.imshow(self.dataset.y_tiles['test'][i])
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')

            classification = self.dataset.y['test'][i]
            if classification[0] == 1:
                classification_result = 'no snake'
            elif classification[1] == 1:
                classification_result = 'cobra'
            elif classification[2] == 1:
                classification_result = 'python'
            else:
                raise Exception

            plt.title('Classification: ' + classification_result)

            if save:
                plt.savefig('{}{}.{}'.format(self.MAIN_DIRECTORY + '/repo/visuals/is_snake_or_not/', str(i), 'png'),
                            bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def plot_rectangle(images, nrows, ncols, directory='', save_file_name='', save_image=False, extension='png',
                       dpi=300):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i, image in enumerate(images[:nrows * ncols]):
            assert isinstance(image, np.ndarray)
            ax[int(np.floor(i / ncols))][int(i % ncols)].imshow(image)
            ax[int(np.floor(i / ncols))][int(i % ncols)].tick_params(labelbottom='off', labelleft='off')
            ax[int(np.floor(i / ncols))][int(i % ncols)].axis('off')
        if save_image is True:
            plt.savefig(save_file_name + '.' + extension, format=extension, dpi=dpi)
        plt.show()

    def plot_single(image, directory, file_name, show=False, extension='png', dpi=300, background_color='white'):
        assert isinstance(image, np.ndarray)
        plt.imshow(image)
        plt.tick_params(labelbottom='off', labelleft='off')
        plt.axis('off')
        plt.rcParams['savefig.facecolor'] = background_color
        plt.savefig('{}{}.{}'.format(directory, file_name, extension),
                    format=extension,
                    dpi=dpi,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        if show:
            plt.show()

    def create_one_row_subplot(*args, names_for_plots=[], file_name='', save_directory=''):
        assert isinstance(names_for_plots, list)
        assert len(names_for_plots) is len(args)
        matplotlib.rcParams.update({'font.size': 4})
        for i, images in enumerate(zip(*args)):
            fig, ax = plt.subplots(nrows=1, ncols=len(images))
            for j, name in enumerate(names_for_plots):
                ax[j].imshow(images[j], vmin=-1, vmax=1)
                ax[j].tick_params(labelbottom='off', labelleft='off')
                ax[j].axis('off')
                ax[j].set_title(name)
                ax[j].colorbar(orientation='vertical')
            plt.savefig('{}{}_{}.{}'.format(save_directory, file_name, i, 'png'),
                        format='png',
                        dpi=300,
                        bbox_inches='tight')
            plt.close()
