import os
import sys
import argparse
import re

import numpy as np

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
sys.path.append(r'/Users/megamot/Programming/Python/Phase-problem-modeling')

from src.propagation.presenter.loader import load_files
import src.propagation.utils.math.units as units
from src.miscellaneous.radius_of_curvature import find_radius
from src.propagation.utils.tie import (
    FFTSolver1D,
    FFTSolver2D,
    DCTSolver1D,
    DCTSolver2D,
    BoundaryConditions,
)

Z_VALUE_PATTERN = r'([-]?\d+\.\d+)\.\w+$'

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
    choices=['fft_1d', 'fft_2d', 'dct_2d', 'dct_1d'],
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
parser.add_argument(
    '--radius',
    type=int,
    default=0,
    help='Включить в метаданные результат расчета радиуса кривизны ВФ'
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
if dz == 0:
    z1 = re.findall(Z_VALUE_PATTERN, args.i1_path)[0]
    z2 = re.findall(Z_VALUE_PATTERN, args.i2_path)[0]
    z1 = units.mm2m(float(z1))
    z2 = units.mm2m(float(z2))
    if z2 > z1:
        dz = z2 - z1
    else:
        dz = z1 - z2
        args.i1_path, args.i2_path = args.i2_path, args.i1_path
args.dz = dz

i1_path = args.i1_path
i2_path = args.i2_path
save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

Solver = args.solver
if Solver == 'fft_1d': Solver = FFTSolver1D
elif Solver == 'fft_2d': Solver = FFTSolver2D
elif Solver == 'dct_1d': Solver = DCTSolver1D
elif Solver == 'dct_2d': Solver = DCTSolver2D
else: raise ValueError(f'There\'s no \"{Solver}\" solver')

bc = args.bc
if bc == 'PBC': bc = BoundaryConditions.PERIODIC
elif bc == 'NBC': bc = BoundaryConditions.NEUMANN
elif bc == 'DBC': bc = BoundaryConditions.DIRICHLET
elif bc == 'None': bc = BoundaryConditions.NONE

print(f'z1: {units.m2mm(z1):>7.3f} mm | z2: {units.m2mm(z2):>7.3f} mm | dz: {units.m2mm(dz):.3f} mm')

# Load Files
intensities = load_files([i1_path, i2_path])
if np.complex in [intensity.dtype for intensity in intensities]:
    raise TypeError('one or more of the intensities has complex dtype')

# TIE
solver = Solver(intensities, dz, wavelength, px_size, bc=bc)
retrieved_phase = solver.solve(args.threshold)

# WaveFront Radius of Curvature
if args.radius:
    radius = find_radius(intensities[1], retrieved_phase, wavelength, px_size)
    args.radius_value = radius

# Сохранение файла с волной
i1_filename = os.path.splitext(os.path.basename(i1_path))[0]
i2_filename = os.path.splitext(os.path.basename(i2_path))[0]
filename = f'TIE {i1_filename} {i2_filename} dz={units.m2mm(dz):.3f}mm.npy'
save_path = os.path.join(save_folder, filename)
np.save(save_path, retrieved_phase)

# Сохранение файла метаданных
filename += '.metadata'
save_path = os.path.join(save_folder, filename)

if len(save_path) > 260:
    raise ValueError(f'Длина имени файла превышает допустимую: len(save_path) = {len(save_path)}; допустимая - 260')

# with open(save_path, 'w') as file:
#     for k, v in vars(args).items():
#         file.write(f'{k}: {v}\n')
