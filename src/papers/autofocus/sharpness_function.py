import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import src.propagation.utils.math.units as units
from src.propagation.presenter.loader import load_file
from src.papers.autofocus.sharpness_functions import sobel_sf, laplace_sf, prewitt_sf, roberts_sf, fft_sf


folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
         r'\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\2. Автофокусировка' \
         r'\z=0.000 propagation std=0.01'

sobel = []
laplace = []
prewitt = []
roberts = []
freq = []
fft = []

dzs = units.um2mm(np.arange(40, 1001, 40))

for dz in dzs:
    # print(f'dz: {dz:.3f} mm')

    z1 = -dz / 2
    z2 = dz / 2

    filename = f'TIE i z={z1:.3f} i z={z2:.3f} dz={dz:.3f}mm.npy'
    filepath = os.path.join(folder, filename)
    phase = load_file(filepath)

    # Убираем нули
    phase -= np.min(phase)

    # Вырезаем центральную область
    height, width = phase.shape
    # clip_w = 200
    # phase = phase[130:160, 490:526]

    # Выделяем границы
    sigma = 1
    low_freq = gaussian_filter(phase, sigma)
    freq_sf = np.sum(phase - low_freq)

    # Считаем SF
    sobel.append(sobel_sf(phase))
    laplace.append(laplace_sf(phase))
    prewitt.append(prewitt_sf(phase))
    roberts.append(roberts_sf(phase))

    window_size = [1010, 1010]
    fft.append(fft_sf(phase, window_size))

    # if dz in map(units.um2mm, [40, 400, 1000]):
    #     plt.figure()
    #     plt.imshow(phase)
    #     plt.colorbar()
    #     plt.title(f'dz: {dz} mm')

    print(f'{sobel[-1]:.0f}\t{laplace[-1]:.0f}\t{prewitt[-1]:.0f}\t{roberts[-1]:.0f}\t{fft[-1]:.0f}')

path = os.path.join(folder, f'sobel.npy')
np.save(path, np.array(sobel))

path = os.path.join(folder, f'laplace.npy')
np.save(path, np.array(laplace))

path = os.path.join(folder, f'prewitt.npy')
np.save(path, np.array(prewitt))

path = os.path.join(folder, f'roberts.npy')
np.save(path, np.array(roberts))

path = os.path.join(folder, f'fft.npy')
np.save(path, np.array(fft))

plt.show()
