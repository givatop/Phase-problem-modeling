import os

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import calculate_radius, calc_amplitude
import src.propagation.utils.math.units as units

plt.style.use('seaborn')

DPI = 100
FIGSIZE = [16, 9]
WAVELENGTH = units.nm2m(515)
WAVENUM = 2 * np.pi / WAVELENGTH
PX_SIZE = units.um2m(5.04)

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\2. Экспериментальные\7. M^2\шаг 500 мкм\3. Для статьи\z=7.0'
filename = 'yslice TIE shifted sigma=25 6.0 shifted sigma=25 8.0 dz=2.000mm.npy'
filepath = os.path.join(folder, filename)
base_filename_1 = os.path.splitext(os.path.basename(filename))[0]

# array = load_file(filepath)[414:1226]  # xslice
array = load_file(filepath)[100:1080-100]  # yslice


if array.ndim == 1:
    degree = 2  # 2 - parabola

    width = array.shape[0]
    x = np.arange(-width//2, width//2) * PX_SIZE

    array -= array.min()

    # min_index = np.argmin(array)
    # shift = min_index - width // 2  # todo нужно по этому значению считать сетку, а не сдвигать её
    # x = np.roll(x, shift)
    #
    # if shift > 0:
    #     x = x[shift:]
    #     array = array[shift:]
    # else:
    #     raise NotImplementedError

    coeffs = np.polyfit(x, array, degree)
    a, b, c = coeffs
    radius = WAVENUM / (2 * a)
    fitted_array = np.poly1d(coeffs)(x)

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI,
                           figsize=FIGSIZE)

    ax.plot(x, array, linestyle='solid', label='Initial')
    # ax.axhline(y=-0.031, linestyle='dashed')

    ax.plot(x, fitted_array, linestyle='dashed', label='Fitted')

    ax.set_title(rf'$ y(x) = {a:.5f}\cdot x^2 + {b:.5f}\cdot x + {c:.5f}. \: R = {radius:.5f} [m]$')
    ax.legend()

    path = os.path.join(folder, f'{base_filename_1} R.png')
    fig.savefig(path)

plt.show()
