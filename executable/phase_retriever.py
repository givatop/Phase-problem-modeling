import os
import sys
import argparse

import numpy as np
from icecream import ic

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
from src.propagation.presenter.loader import load_files
from src.propagation.utils.math.units import m2mm
from src.propagation.utils.tie import (
    FFTSolver,
    FFTSolver1D,
    BoundaryConditions,
    SimplifiedFFTSolver,
    SimplifiedFFTSolver1D,
)

parser = argparse.ArgumentParser(description='Retrieve Phase via Transport-of-Intensity Equation')

# region Физические параметры волны
parser.add_argument(
    '--wavelength',
    type=float,
    required=True,
    help='Длина волны'
)
parser.add_argument(
    '--px_size',
    type=float,
    required=True,
    help='Размер пикселя матрицы приемника излучения'
)
parser.add_argument(
    '--dz',
    type=float,
    required=True,
    help='Расстояние между плоскостями I1 и I2'
)
parser.add_argument(
    '--i1_path',
    type=str,
    required=True,
    help='Путь к .npy или графическому файлу с 1-й интенсивностью'
)
parser.add_argument(
    '--i2_path',
    type=str,
    required=True,
    help='Путь к .npy или графическому файлу со 2-й интенсивностью'
)
# endregion
# region Параметры TIE
parser.add_argument(
    '--solver',
    type=str,
    choices=['fft_1d', 'fft_2d', 'simplified_fft1d', 'simplified_fft2d', 'dct_2d'],
    default='fft_2d',
    required=False,
    help='Метод решения TIE'
)
parser.add_argument(
    '--bc',
    type=str,
    choices=['PBC', 'NBC', 'DBC', 'None'],
    default='None',
    required=False,
    help='Тип граничных условий'
)
parser.add_argument(
    '--threshold',
    type=float,
    required=True,
    help='Все значения ниже этого порога будут приравнены к порогу'
)
# endregion
# region Параметры сохранения
parser.add_argument(
    '--save_folder',
    type=str,
    required=True,
    help='Путь к папке, куда будут сохраняться файлы'
)
# endregion
# region Парсинг в переменные и вывод в консоль
args = parser.parse_args()
wavelength = args.wavelength
px_size = args.px_size
dz = args.dz
i1_path = args.i1_path
i2_path = args.i2_path
save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

Solver = args.solver
if Solver == 'fft_2d': Solver = FFTSolver
elif Solver == 'fft_1d': Solver = FFTSolver1D
elif Solver == 'simplified_fft1d': Solver = SimplifiedFFTSolver1D
elif Solver == 'simplified_fft2d': Solver = SimplifiedFFTSolver
elif Solver == 'dct_2d': raise NotImplementedError
else: raise ValueError(f'There\'s no \"{Solver}\" solver')

bc = args.bc
if bc == 'PBC': bc = BoundaryConditions.PERIODIC
elif bc == 'NBC': bc = BoundaryConditions.NEUMANN
elif bc == 'DBC': bc = BoundaryConditions.DIRICHLET
elif bc == 'None': bc = BoundaryConditions.NONE

threshold = args.threshold

# todo DEBUGGING
# wavelength = 555e-6
# px_size = 5e-6
# dz = 1e-3
# i1_path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. Проверка корректности FFT1d-решения\phi=sphere i=gauss 1D complex_field propagation\intensity z = 0.000.npy'
# i2_path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. Проверка корректности FFT1d-решения\phi=sphere i=gauss 1D complex_field propagation\intensity z = 10.000.npy'
# save_folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. Проверка корректности FFT1d-решения\phi=sphere i=gauss 1D complex_field propagation'
# Solver = FFTSolver1D
# bc = BoundaryConditions.NONE
# threshold = .1

ic(args)
ic(wavelength)
ic(px_size)
ic(dz)
ic(i1_path)
ic(i2_path)
ic(save_folder)
ic(Solver)
ic(bc)
ic(threshold)
# endregion

# Load Files
intensities = load_files([i1_path, i2_path])

# TIE
solver = Solver(intensities, dz, wavelength, px_size, bc=bc)
retrieved_phase = solver.solve(threshold)

# Сохранение файла с волной
i1_filename = os.path.splitext(os.path.basename(i1_path))[0]
i2_filename = os.path.splitext(os.path.basename(i2_path))[0]
filename = f'TIE {i1_filename} {i2_filename} dz={m2mm(dz):.3f}mm.npy'
save_path = os.path.join(save_folder, filename)
ic(save_path)
np.save(save_path, retrieved_phase)

# Сохранение файла метаданных
filename += '.metadata'
save_path = os.path.join(save_folder, filename)
with open(save_path, 'a') as file:
    for k, v in vars(args).items():
        file.write(f'{k}: {v}\n')
    file.write(f'dz: {m2mm(dz):.3f} mm\n')
