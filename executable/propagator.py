import os
import sys
import argparse

import numpy as np
from icecream import ic

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
from src.propagation.utils.math.units import m2mm
from src.propagation.utils.optic.propagation_methods import (
    angular_spectrum_band_limited,
    angular_spectrum_propagation,
    fresnel
)

parser = argparse.ArgumentParser(description='Propagate initial wave on desired distances')

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
    '--wave_path',
    type=str,
    required=True,
    help='Путь к .npy-файлу с волной'
)
# endregion
# region Параметры распространения
parser.add_argument(
    '--start',
    type=float,
    required=True,
    help='z = 0 (м)'
)
parser.add_argument(
    '--stop',
    type=float,
    required=True,
    help='z = z (м)'
)
parser.add_argument(
    '--step',
    type=float,
    required=True,
    help='dz (м)'
)
parser.add_argument(
    '--method',
    type=str,
    choices=['angular_spectrum_band_limited', 'angular_spectrum', 'fresnel'],
    default='angular_spectrum_band_limited',
    required=False,
    help='Метод распространения'
)
# endregion
# region Параметры сохранения
parser.add_argument(
    '--save_folder',
    type=str,
    required=True,
    help='Путь к папке, куда будут сохраняться файлы'
)
parser.add_argument(
    '--separate_save',
    type=int,
    default=1,
    help='Созранить интенсивность и фазу как отдельные файлы'
)
# endregion
# region Парсинг в переменные и вывод в консоль
args = parser.parse_args()
wavelength = args.wavelength
px_size = args.px_size
wave_path = args.wave_path
start = args.start
stop = args.stop
step = args.step
method = args.method
save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

complex_field = np.load(wave_path)

# Grid
if complex_field.ndim == 1:
    ONE_DIMENSION = True
    height, width = 1, complex_field.shape[0]
elif complex_field.ndim == 2:
    height, width = complex_field.shape
else:
    ValueError(f'Unknown shape: {complex_field.shape}')

distances = np.arange(start, stop + step, step)
if method == 'fresnel': method = fresnel
elif method == 'angular_spectrum': method = angular_spectrum_propagation  # todo это не будет работать
elif method == 'angular_spectrum_band_limited': method = angular_spectrum_band_limited

ic(wavelength)
ic(px_size)
ic(wave_path)
ic(list(distances))
ic(method)
ic(save_folder)
ic(height, width)
# endregion

for distance in distances:
    # Консольный вывод
    ic(distance)

    # Распространение
    wave_z = method(complex_field, distance, wavelength, px_size)

    # Сохранение файла с волной
    filename = f'z={m2mm(distance):.3f}.npy'
    save_path = os.path.join(save_folder, filename)
    print(save_path)
    np.save(save_path, wave_z)

    # Раздельное сохранение интенсивности и фазы
    intensity = np.abs(wave_z) ** 2
    filename = f'intensity z = {m2mm(distance):.3f}.npy'
    save_path = os.path.join(save_folder, filename)
    np.save(save_path, intensity)
    phase = np.unwrap(np.angle(wave_z))
    filename = f'phase z = {m2mm(distance):.3f}.npy'
    save_path = os.path.join(save_folder, filename)
    np.save(save_path, phase)

    # Сохранение файла метаданных.
    filename = f'z = {m2mm(distance):.3f}.npy.metadata'
    save_path = os.path.join(save_folder, filename)
    with open(save_path, 'a') as file:
        for k, v in vars(args).items():
            file.write(f'{k}: {v}\n')
        file.write(f'distance: {m2mm(distance):.3f}\n')
