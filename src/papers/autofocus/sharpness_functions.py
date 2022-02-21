import os
import numpy as np
from scipy.ndimage import convolve
from src.propagation.presenter.loader import load_file
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.propagation.utils.math.units as units


def create_cbar(ax, img, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel(label)
    return cbar


def fft_sf(array, window_size):
    height, width = array.shape
    window_height, window_width = window_size

    return np.sum(np.abs(np.fft.fft2(array)[
                         height // 2 - window_height // 2:height // 2 + window_height // 2,
                         width // 2 - window_width // 2:width // 2 + window_width // 2]))


def roberts_sf(array):
    kernelx = np.array([
        [-1, 0],
        [ 0, 1]
    ])
    kernely = np.array([
        [0, -1],
        [1,  0]
    ])

    x = np.abs(convolve(array, kernelx))
    y = np.abs(convolve(array, kernely))

    return np.sum(np.sqrt(x ** 2 + y ** 2))


def sobel_sf(array):
    kernelx = np.array([
        [+1, 0, -1],
        [+2, 0, -2],
        [+1, 0, -1],
    ])
    kernely = np.array([
        [+1, +2, +1],
        [ 0,  0,  0],
        [-1, -2, -1],
    ])

    x = np.abs(convolve(array, kernelx))
    y = np.abs(convolve(array, kernely))

    return np.sum(np.sqrt(x ** 2 + y ** 2))


def prewitt_sf(array):
    kernelx = np.array([
        [+1, 0, -1],
        [+1, 0, -1],
        [+1, 0, -1],
    ])
    kernely = np.array([
        [+1, +1, +1],
        [ 0,  0,  0],
        [-1, -1, -1],
    ])

    x = np.abs(convolve(array, kernelx))
    y = np.abs(convolve(array, kernely))

    return np.sum(np.sqrt(x ** 2 + y ** 2))


def laplace_sf(array, inward=True):
    if inward:
        kernel = np.array([
            [0, +1, 0],
            [1, -4, 1],
            [0, +1, 0],
        ])
    else:
        kernel = np.array([
            [0,  -1, 0],
            [-1, +4, -1],
            [0,  -1, 0],
        ])

    convolved = np.abs(convolve(array, kernel))

    return np.sum(np.abs(convolved))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dpi = 100
    figsize = [8.4, 4.8]

    folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
             r'\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\2. Автофокусировка\z=0.000 propagation std=0.01'

    dzs = units.um2mm(np.arange(40, 51, 80))

    for dz in dzs:

        z1 = -dz / 2
        z2 = dz / 2

        filename = f'TIE i z={z1:.3f} i z={z2:.3f} dz={dz:.3f}mm.npy'

        origin = 'BMSTU -b 1024x1024.png'
        origin = load_file(os.path.join(folder, origin))

        filepath = os.path.join(folder, filename)
        image = load_file(filepath)

        kernel = np.array([  # laplace
            [0, +1, 0],
            [1, -4, 1],
            [0, +1, 0],
        ])

        convolved = np.abs(convolve(image, kernel))
        sf = np.sum(convolved)

        anti_sf = np.sum(np.abs(image - convolved))

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, dpi=dpi, figsize=figsize)

        im1 = ax1.imshow(origin[640:713, 431:463])
        im2 = ax2.imshow(image[640:713, 431:463])
        im3 = ax3.imshow(convolved[640:713, 431:463])

        create_cbar(ax1, im1)
        create_cbar(ax2, im2)
        create_cbar(ax3, im3)

        ax1.set_title(f'Initial Phase')
        ax2.set_title(f'TIE')
        ax3.set_title(f'Laplacian')

        # fig.suptitle(f'SF = {sf:.0f}')
        print(f'{dz:.3f}\t{sf:.0f}\t{anti_sf:.0f}')

        fig.tight_layout()

    plt.show()
