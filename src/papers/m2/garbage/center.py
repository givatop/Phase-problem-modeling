import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import normalize
import src.propagation.utils.optic as optic

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\М2\12.02'

DPI = 150
FIGSIZE = [6.4, 4.8]

distances = np.arange(1, 2)

for distance in distances:

    filename = f'{distance:.1f}.tif'
    path = os.path.join(folder, filename)
    array = load_file(path)
    # array[array < array.max() * .5] = 0
    # array = np.roll(array, [800, 100], [1, 0])

    x = np.arange(-50, 50)
    y = np.arange(-50, 50)
    X, Y = np.meshgrid(x, y)
    # array = optic.gauss_2d(X, Y, wx=10, wy=10, x0=25, y0=-25)

    # Dull energy center
    dull_row, dull_col = np.unravel_index(np.argmax(array), array.shape)

    # Smart energy center
    smart_col = np.sum(np.sum(array, axis=0) *
                np.array(np.arange(1, array.shape[1]+1))) / \
                np.sum(array)
    smart_col = int(np.ceil(smart_col - 1))

    smart_row = np.sum(np.sum(array, axis=1) *
                np.array(np.arange(1, array.shape[0]+1))) / \
                np.sum(array)
    smart_row = int(np.ceil(smart_row - 1))


    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI,
                           figsize=FIGSIZE)

    ax.imshow(array)

    # ax.axhline(y=dull_row, linestyle='dotted', color='red')
    # ax.axvline(x=dull_col, linestyle='dotted', color='red')

    ax.axhline(y=607, linestyle='dotted', color='green')
    ax.axvline(x=1345, linestyle='dotted', color='green')

    print(dull_col, dull_row)
    print(smart_col, smart_row)

    # filename = f'f {distance:.1f}.tif'
    # path = os.path.join(folder, filename)

plt.show()