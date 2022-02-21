import os

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter, median_filter
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import normalize
from src.propagation.utils.math.general import row_slice, col_slice
import src.propagation.utils.math.units as units
import src.propagation.utils.optic as optic

plt.style.use('seaborn')
folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\2. Экспериментальные\7. M^2\шаг 500 мкм'

DPI = 150
FIGSIZE = [6.4, 4.8]
CMAP = "jet"
VMIN = 0
VMAX = .8
PRECISE = 1
STEP = 1
SIGMA = 25
MEDIAN_SIZE = 15
PX_SIZE = units.um2mm(5.04)

distances = np.arange(0, 10.1, 1)


for distance in distances:
    # filename = f'{distance:.{PRECISE}f}.tif'
    filename = f'median_{MEDIAN_SIZE} gauss_{SIGMA} {distance:.{PRECISE}f}.npy'
    path = os.path.join(folder, filename)

    intensity = load_file(path)

    height, width = intensity.shape
    x = np.arange(-width // 2, width // 2) * PX_SIZE
    y = np.arange(-height // 2, height // 2) * PX_SIZE
    X, Y = np.meshgrid(x, y)

    diameter = 950 * PX_SIZE
    # row, col = np.unravel_index(np.argmax(intensity), intensity.shape)
    row, col = 585, 795  # значения подобраны эмпирически
    x0 = (col - width // 2) * PX_SIZE
    y0 = (row - height // 2) * PX_SIZE
    aperture = optic.circ_cartesian(X, Y, w=diameter, x0=x0, y0=y0)

    # intensity = intensity * aperture

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)

    img = ax.imshow(intensity, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel('a.u.')

    ax.axvline(x=col)
    ax.axhline(y=row)

    fig.tight_layout()

    path = os.path.join(folder, f'apd {filename[:-4]}.png')
    fig.savefig(path)

    # New
    new_width = width * 2
    new_height = height * 2
    new_intensity = np.zeros([new_height, new_width])

    col_left  = new_width // 2 - width // 2
    col_right = new_width // 2 + width // 2

    row_left  = new_height // 2 - height // 2
    row_right = new_height // 2 + height // 2

    new_intensity[row_left:row_right, col_left:col_right] = intensity

    path = os.path.join(folder, f'new no apd {filename}')
    np.save(path, new_intensity)

    print(f'z: {distance:>4.1f}; max: {np.max(intensity):.3f}; col: {col:>4}; row: {row:> 4}')

    # todo delete
    fig.clear()

# plt.show()
