import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.propagation.utils.math import units
from src.propagation.utils.tie import FFTSolver, BoundaryConditions
from src.propagation.utils.math.units import (nm2m, mm2m, um2m, m2mm, m2um, percent2decimal)
from src.propagation.presenter.loader import load_image


# Путь к папке с файлами
base_folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\2. Исследования\2021 TIE Алмаз'
concrete_folder = 'CaF2 White LED Reflection Повтор'
filepath = os.path.join(base_folder, concrete_folder)

# основные параметры для синтеза волны
width, height = 2592, 1944
wavelength = units.nm2m(555)
px_size = units.um2m(7.4)  # todo учесть увеличение!!!
bc = BoundaryConditions.NONE
threshold = np.exp(1) ** -2

# Параметры сеток
z_shift = um2m(50)
dzs = list(map(um2m, [5, 10, 15, 20]))
z1_start, z1_stop = um2m(90), um2m(115)
cut_step = 4
a, b = 0, 0

# параметры для формирования имени папки с данными
# start = units.mm2m(0)
# stop = units.mm2m(1)
# step = units.mm2m(1)

for dz in dzs:
    print(m2um(dz))
    # folder_name = f'z_{units.m2mm(start)}-{units.m2mm(stop)}-{units.m2mm(step)} {width}x{height}'

    # Создание сеток
    z2_start, z2_stop = z1_start + dz, z1_stop + dz
    z1_list = [current_z for current_z in np.arange(z1_start, z1_stop, z_shift)]
    z2_list = [current_z for current_z in np.arange(z2_start, z2_stop, z_shift)]

    for z1, z2 in zip(z1_list, z2_list):
        fn1 = os.path.join(filepath, f'{m2um(z1):.0f}.JPG')
        fn2 = os.path.join(filepath, f'{m2um(z2):.0f}.JPG')
        paths = [fn1, fn2]
        intensities = [load_image(path) for path in paths]

        Y, X = np.mgrid[0:intensities[0].shape[0], 0:intensities[0].shape[1]]

        solver = FFTSolver(intensities, dz, wavelength, px_size, bc)
        unwrapped_phase = solver.solve(threshold)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(
            X[
                ::cut_step, ::cut_step
                # (unwrapped_phase.shape[0] // 2 - b // 2):(unwrapped_phase.shape[0] // 2 + b // 2):cut_step,
                # (unwrapped_phase.shape[1] // 2 - a // 2):(unwrapped_phase.shape[1] // 2 + a // 2):cut_step
            ],
            Y[
                ::cut_step, ::cut_step
                # (unwrapped_phase.shape[0] // 2 - b // 2):(unwrapped_phase.shape[0] // 2 + b // 2):cut_step,
                # (unwrapped_phase.shape[1] // 2 - a // 2):(unwrapped_phase.shape[1] // 2 + a // 2):cut_step
            ],
            unwrapped_phase[
                ::cut_step, ::cut_step
                # (unwrapped_phase.shape[0] // 2 - b // 2):(unwrapped_phase.shape[0] // 2 + b // 2):cut_step,
                # (unwrapped_phase.shape[1] // 2 - a // 2):(unwrapped_phase.shape[1] // 2 + a // 2):cut_step
            ]
        )
        fig2 = plt.figure()
        ax2 = fig2.gca()
        im = ax2.imshow(unwrapped_phase)

        # colorbar
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()
        # plt.close(fig)
