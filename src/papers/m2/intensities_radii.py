import os

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter, median_filter
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import normalize, energy_center
from src.propagation.utils.math.general import row_slice, col_slice
import src.propagation.utils.math.units as units

plt.style.use('seaborn')
folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\М2\data\12.09'

DPI = 150
FIGSIZE = [8.4, 4.8]
colors = ['magenta', 'green', 'blue', 'red', 'orange', 'cyan', 'magenta',
          'green', 'blue', 'red', 'orange', 'cyan', 'magenta', 'green', 'blue', 'red', 'orange', 'cyan', 'green', 'blue', 'red', 'orange', 'cyan', 'magenta', 'green', 'blue', 'red', 'orange', 'cyan', ]

SIGMA = 0.8
PX_SIZE_MM = units.um2mm(5.04)

wxs, wys = [], []

distances = np.arange(0, 143)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=DPI, figsize=FIGSIZE)

with open(os.path.join(folder, f'intensities radii.txt'), mode='wt') as txt:
    txt.write(f'sigma: {SIGMA}\n')
    txt.write(f'xmax: {1920 * PX_SIZE_MM:.3f} mm | ymax: {1080 * PX_SIZE_MM:.3f} mm\n\n')
    txt.write(f'{"z":>2} | {"wx":^7} | {"wy":^7}\n')

print(f'sigma: {SIGMA}')
print(f'xmax: {1920 * PX_SIZE_MM:.3f} mm | ymax: {1080 * PX_SIZE_MM:.3f} mm ')

for distance in distances:

    filename = f'f {distance:.1f}.tif'
    path = os.path.join(folder, filename)
    array = load_file(path)

    # С фильтром
    array_filt = gaussian_filter(array, sigma=(SIGMA, SIGMA))

    dull_row, dull_col = np.unravel_index(np.argmax(array_filt), array.shape)

    ax1.imshow(array_filt, cmap='jet')
    ax1.axhline(y=dull_row, linestyle='dotted', color='red')
    ax1.axvline(x=dull_col, linestyle='dotted', color='red')

    # Сечение
    rowslice = row_slice(array_filt, dull_row)
    colslice = col_slice(array_filt, dull_col)

    ax2.plot(colslice)
    ax2.plot(rowslice)

    assert np.max(rowslice) == np.max(colslice)

    threshold = np.exp(-2) * np.max(colslice)
    col_mask = colslice >= threshold
    row_mask = rowslice >= threshold
    wx = np.max(np.sum(row_mask)) * PX_SIZE_MM / 2
    wy = np.max(np.sum(col_mask)) * PX_SIZE_MM / 2

    wxs.append(wx)
    wys.append(wy)

    with open(os.path.join(folder, f'intensities radii.txt'), mode='at') as txt:
        wx_str = f'{wx:.3f}'.replace('.', ',')
        wy_str = f'{wy:.3f}'.replace('.', ',')
        txt.write(f'{distance}\t{wx_str}\t{wy_str}\n')

    filename = f'sigma={SIGMA} f {distance:.1f}.npy'
    path = os.path.join(folder, filename)
    np.save(path, array_filt)

    print(f'z\t{distance:>4}\twx\t{wx:.3f}\twy\t{wy:.3f}')


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)

ax.plot(distances, wxs, label='x-radius')
ax.plot(distances, wys, label='y-radius')
ax.legend()
ax.set_title('Intensities Radii')
ax.set_xlabel('z, mm')
ax.set_ylabel('radius, mm')

path = os.path.join(folder, f'intensities radii.png')
fig.savefig(path)
plt.close(fig)

# plt.show()
