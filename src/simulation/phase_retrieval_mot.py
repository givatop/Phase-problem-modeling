import os

import matplotlib.pyplot as plt
import numpy as np

from src.propagation.presenter.loader import load_files
from src.propagation.utils.math import units
from src.propagation.utils.tie import FFTSolver1D, BoundaryConditions, FFTSolver2D

# основные параметры для синтеза волны
from src.propagation.utils.tie.dct_solver import DCTSolver2D, DCTSolver1D

bc = BoundaryConditions.NONE
threshold = np.exp(1) ** -2
width, height = 512, 512
wavelength = units.nm2m(555)
px_size = units.um2m(5)
gaussian_width_params = [250]
focal_lens = [100]
focal_lens = list(map(units.mm2m, focal_lens))

# Параметры сеток
z_shift = 0.001
dzs = [1]
z1_start, z1_stop = -0.001, 0.001

# параметры для формирования имени папки с данными
start = units.mm2m(0.001)
stop = units.mm2m(-0.001)
step = units.mm2m(0.001)

for focal_len in focal_lens:
    for gaussian_width_param in gaussian_width_params:
        for dz in dzs:

            # folder_name = \
            #     f'z_{units.m2mm(start)}-{units.m2mm(stop)}-{units.m2mm(step)} ' \
            #     f'f_{units.m2mm(focal_len)} ' \
            #     f'w_{gaussian_width_param} ' \
            #     f'{width}x{height}'
            #
            # filepath = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir,
            #                         'data', folder_name, 'intensity npy')

            # filepath = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data',
            #                         'executable_props', '19201')

            filepath = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data',
                                    'z_-0.005-0.005-0.001 f_100.0 w_250 512x512', 'intensity npy')

            # Создание сеток
            # z2_start, z2_stop = z1_start + dz, z1_stop + dz
            # z1_list = [current_z for current_z in np.arange(z1_start, z1_stop, z_shift)]
            # z2_list = [current_z for current_z in np.arange(z2_start, z2_stop, z_shift)]

            # for z1, z2 in zip(z1_list, z2_list):
            fn1 = os.path.join(filepath, f'z_{z1_start:.3f}mm.npy')
            fn2 = os.path.join(filepath, f'z_{z1_stop:.3f}mm.npy')
            # fn1 = os.path.join(filepath, 'i z=0.001.npy')
            # fn2 = os.path.join(filepath, 'i z=-0.001.npy')
            paths = [fn1, fn2]
            intensities = load_files(paths)

            solver = DCTSolver2D(intensities, dz, wavelength, px_size, bc)
            # unwrapped_phase = solver.solve(threshold)
            unwrapped_phase = solver.solve(threshold)
            plt.imshow(unwrapped_phase, cmap='jet')
            plt.colorbar()
            plt.show()
            pass

