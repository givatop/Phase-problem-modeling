import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import row_slice, col_slice
import src.propagation.utils.math.units as units

plt.style.use('seaborn')
folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\М2\data\12.09'

DPI = 150
FIGSIZE = [8.4, 5.8]
colors = ['magenta', 'green', 'blue', 'red', 'orange', 'cyan', 'magenta',]

SIGMA = 0.8
distances = np.arange(0, 101, 10)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=DPI, figsize=FIGSIZE)

for (distance, color) in zip(distances, colors):
    print(f'z: {distance:.3f} mm')

    filename = f'f {distance:.1f}.tif'
    path = os.path.join(folder, filename)
    intensity = load_file(path)

    # Фильтрация
    intensity = gaussian_filter(intensity, sigma=(SIGMA, SIGMA))

    # Сечения
    height, width = intensity.shape

    # Центрирование
    row, col = np.unravel_index(np.argmax(intensity), intensity.shape)
    x_shift = -col + width // 2
    y_shift = -row + height // 2
    intensity = np.roll(intensity, (y_shift, x_shift), axis=(0, 1))

    # Сечения
    row, col = np.unravel_index(np.argmax(intensity), intensity.shape)
    rowslice = row_slice(intensity, row)
    colslice = col_slice(intensity, col)

    # Графики
    index_radius = 40
    left = -index_radius + width // 2
    right = index_radius + width // 2

    ax1.plot(rowslice[left:right], marker='', label=f'z = {distance:.0f} mm',
             linestyle='solid')

    # ax1.axvline(x=np.argmax(rowslice[left:right]), linestyle='dotted', color=color)

    left = -index_radius + height // 2
    right = index_radius + height // 2

    ax2.plot(colslice[left:right], marker='', label=f'z = {distance:.0f} mm',
             linestyle='solid')

    # ax2.axvline(x=np.argmax(colslice[left:right]), linestyle='dotted', color=color)


ax1.set_xlabel('x, mm')
ax1.legend()
ax1.set_ylabel('intensity, a.u.')
ax1.set_ylim([-0.1, 0.9])

ax2.set_xlabel('y, mm')
ax2.legend()
ax2.set_ylabel('intensity, a.u.')
ax2.set_ylim([-0.1, 0.9])

fig.tight_layout()

# path = os.path.join(folder, f'intensities slices sigma={SIGMA:.1f}.png')
# fig.savefig(path)

# path = os.path.join(folder, f'intensities slices.svg')
# fig.savefig(path)

plt.show()
