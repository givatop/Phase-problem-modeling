import os

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from src.propagation.presenter.loader import load_files

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\3. Fresnel\1. Сф. волна f=10cm'
filename_1 = 'z=0.000.txt'
filename_2 = 'z=0.005.txt'
filepath_1 = os.path.join(folder, filename_1)
filepath_2 = os.path.join(folder, filename_2)
base_filename_1 = os.path.splitext(os.path.basename(filename_1))[0]
base_filename_2 = os.path.splitext(os.path.basename(filename_2))[0]

i1, i2 = load_files([filepath_1, filepath_2])

fig4 = plt.figure(dpi=100, figsize=[16, 9])

if i1.ndim == 2:
    ax41, ax42, ax43, ax44 = fig4.add_subplot(2, 2, 1), \
                             fig4.add_subplot(2, 2, 2), \
                             fig4.add_subplot(2, 2, 3), \
                             fig4.add_subplot(2, 2, 4)

    step = 1
    abs_max = np.argmax(i1)
    y_max, x_max = np.unravel_index(abs_max, i1.shape)
    i1_yslice = i1[::step, x_max]
    i2_yslice = i2[::step, x_max]
    i1_xslice = i1[y_max, ::step]
    i2_xslice = i2[y_max, ::step]

    polynomial_order = 3
    window_height = 151
    mode = 'interp'
    i1_xslice_filtered = savgol_filter(i1_xslice, window_height, polynomial_order, mode=mode)
    i2_xslice_filtered = savgol_filter(i2_xslice, window_height, polynomial_order, mode=mode)
    i1_yslice_filtered = savgol_filter(i1_yslice, window_height, polynomial_order, mode=mode)
    i2_yslice_filtered = savgol_filter(i2_yslice, window_height, polynomial_order, mode=mode)

    ax41.plot(i1_xslice_filtered, label=filename_1)
    ax41.plot(i2_xslice_filtered, label=filename_2)
    ax42.plot(i1_yslice_filtered, label=filename_1)
    ax42.plot(i2_yslice_filtered, label=filename_2)
    ax43.plot(i2_xslice_filtered - i1_xslice_filtered)
    ax44.plot(i2_yslice_filtered - i1_yslice_filtered)

    [ax.legend() for ax in [ax41, ax42]]
    [ax.grid() for ax in [ax41, ax42, ax43, ax44]]
    ax41.set_title(f'({x_max}, {y_max})')

    fp = os.path.join(
        folder,
        f'slices '
        f'{os.path.splitext(os.path.basename(filename_1))[0]} '
        f'{os.path.splitext(os.path.basename(filename_2))[0]} '
        f'{polynomial_order} {window_height} {mode}'
        f'.tif'
    )
    fig4.tight_layout()
    fig4.savefig(fp, bbox_inches='tight', pad_inches=0.1)
    np.save(os.path.join(folder, f'xslice {base_filename_1}.npy'), i1_xslice_filtered)
    np.save(os.path.join(folder, f'xslice {base_filename_2}.npy'), i2_xslice_filtered)
    np.save(os.path.join(folder, f'yslice {base_filename_1}.npy'), i1_yslice_filtered)
    np.save(os.path.join(folder, f'yslice {base_filename_2}.npy'), i2_yslice_filtered)
elif i1.ndim == 1:
    ax41, ax42 = fig4.add_subplot(2, 1, 1), fig4.add_subplot(2, 1, 2)

    step = 1
    i1_xslice = i1[::step]
    i2_xslice = i2[::step]

    polynomial_order = 3
    window_height = 151
    mode = 'interp'
    i1_xslice_filtered = savgol_filter(i1_xslice, window_height, polynomial_order, mode=mode)
    i2_xslice_filtered = savgol_filter(i2_xslice, window_height, polynomial_order, mode=mode)

    ax41.plot(i1_xslice_filtered, label=filename_1)
    ax41.plot(i2_xslice_filtered, label=filename_2)
    ax42.plot(i2_xslice_filtered - i1_xslice_filtered)

    [ax.legend() for ax in [ax41]]
    [ax.grid() for ax in [ax41, ax42]]

    fp = os.path.join(
        folder,
        f'slices '
        f'{os.path.splitext(os.path.basename(filename_1))[0]} '
        f'{os.path.splitext(os.path.basename(filename_2))[0]} '
        f'{polynomial_order} {window_height} {mode}'
        f'.tif'
    )
    fig4.tight_layout()
    fig4.savefig(fp, bbox_inches='tight', pad_inches=0.1)
    np.save(os.path.join(folder, f'xslice {base_filename_1}.npy'), i1_xslice_filtered)
    np.save(os.path.join(folder, f'xslice {base_filename_2}.npy'), i2_xslice_filtered)

ic(filename_1)
ic(filename_2)
