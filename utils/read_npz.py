import numpy as np
import matplotlib.pyplot as plt

from utils.plotter import Plotter

if __name__ == '__main__':
    file_path = '../repo/datasets/snakes_results.npz'
    loaded = np.load(file_path)
    for key in loaded.keys():
        print(key + ': ' + str(loaded[key].shape))

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

    plot_rectangle(loaded['king_cobra'], 2, 5)
    print(np.max(loaded['king_cobra'][0]))