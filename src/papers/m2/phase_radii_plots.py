import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.propagation.utils.math.units as units


plt.style.use('seaborn')
DPI = 100
FIGSIZE = [6.4, 4.8]
CMAP = 'jet'


def create_cbar(ax, img, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel(label)
    return cbar


dzs = [60, 70, 80, 90, 100, 110, 120]
start = 0
stop = 143

for dz in dzs:
    folder = rf'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\1. Работа\ФИАН\06. TIE\1. М2\5. Эксперименты\12.09\dz={dz}'

    # Загрузка функций резкости
    path = os.path.join(folder, f'radii dz={dz:.0f} mm zs.npy')
    zs = np.load(path)[start:stop]

    path = os.path.join(folder, f'radii dz={dz:.0f} mm xradii.npy')
    xradii = np.load(path)[start:stop]

    path = os.path.join(folder, f'radii dz={dz:.0f} mm yradii.npy')
    yradii = np.load(path)[start:stop]

    # Графики
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)

    ax.plot(zs, xradii, label='x-radii')
    ax.plot(zs, yradii, label='y-radii')

    ax.set_xlabel('z, mm')
    ax.set_ylabel('Wavefront Radii, mm')
    ax.set_title(f'dz = {dz} mm')
    ax.legend()

    path = os.path.join(folder, f'radii dz={dz} mm.png')
    fig.savefig(path)

    plt.show()
    plt.close(fig)
