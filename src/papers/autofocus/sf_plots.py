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


folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
         r'\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\2. Автофокусировка' \
         r'\z=0.000 propagation std=0.01'

# dz для оси Х
dzs = units.um2mm(np.arange(40, 1001, 40))

# Загрузка функций резкости
path = os.path.join(folder, f'sobel.npy')
sobel = np.load(path)
sobel -= np.min(sobel)
sobel_peak = dzs[np.argmax(sobel)]

path = os.path.join(folder, f'laplace.npy')
laplace = np.load(path)
laplace -= np.min(laplace)
laplace_peak = dzs[np.argmax(laplace)]

path = os.path.join(folder, f'prewitt.npy')
prewitt = np.load(path)
prewitt -= np.min(prewitt)
prewitt_peak = dzs[np.argmax(prewitt)]

path = os.path.join(folder, f'roberts.npy')
roberts = np.load(path)
roberts -= np.min(roberts)
roberts_peak = dzs[np.argmax(roberts)]

path = os.path.join(folder, f'fft.npy')
fft = np.load(path)
fft -= np.min(fft)
fft_peak = dzs[np.argmax(fft)]

# Графики
fig, ax = plt.subplots(ncols=1, nrows=1, dpi=DPI, figsize=FIGSIZE)

ax.plot(dzs, sobel, label=f'Sobel. dz={sobel_peak}')
ax.plot(dzs, laplace, label=f'Laplace. dz={laplace_peak}')
ax.plot(dzs, prewitt, label=f'Prewitt. dz={prewitt_peak}')
ax.plot(dzs, roberts, label=f'Roberts. dz={roberts_peak}')
ax.plot(dzs, fft, label=f'FFT. dz={fft_peak}')

ax.legend()
ax.set_xlabel(f'dz, mm')
ax.set_ylabel(f'normilized SF, a.u.')
ax.set_title(f'Sharpness Function')
ax.set_yscale('log')

path = os.path.join(folder, f'SF.png')
fig.savefig(path)

# plt.show()
