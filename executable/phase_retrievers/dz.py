import os
import sys

import numpy as np
from icecream import ic

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
from src.propagation.presenter.loader import load_files
from src.propagation.utils.math.units import m2mm, mm2m
from src.propagation.utils.tie import (
    FFTSolver2D,
    FFTSolver1D,
    BoundaryConditions,
    SimplifiedFFTSolver,
    SimplifiedFFTSolver1D,
)

wavelength = 555e-9
px_size = 5e-6
intensity_folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. FFT 1D\i=gauss phi=sphere propagation'
save_folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. FFT 1D\i=gauss phi=sphere propagation\dz check'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
Solver = FFTSolver1D
bc = BoundaryConditions.NONE
threshold = .1

# Параметры сеток, мм
distances = [mm2m(num) for num in np.arange(0, 11, 10)]
z1 = distances.pop(0)


for z2 in distances:
    dz = z2 - z1

    fn1, fn2 = f'i z={m2mm(z1):.3f}.npy', f'i z={m2mm(z2):.3f}.npy'
    fp1, fp2 = os.path.join(intensity_folder, fn1), os.path.join(intensity_folder, fn2)
    intensities = load_files([fp1, fp2])

    # TIE
    ic(dz)
    solver = Solver(intensities, dz, wavelength, px_size, bc=bc)
    retrieved_phase = solver.solve(threshold)

    # Сохранение файла с волной
    filename = f'TIE {fn1[:-4]} {fn2[:-4]} mm.npy'
    # filename = f'TIE {fn1[:-4]} {fn2[:-4]} dz={m2mm(dz):.3f}mm.npy'
    save_path = os.path.join(save_folder, filename)
    ic(save_path)
    np.save(save_path, retrieved_phase)

    # Сохранение файла метаданных
    filename += '.metadata'
    save_path = os.path.join(save_folder, filename)
    metadata = {
        'wavelength, m': wavelength,
        'px_size, m': px_size,
        'intensity_folder': intensity_folder,
        'save_folder': save_folder,
        'Solver': Solver,
        'bc': bc,
        'threshold': threshold,
        'z1, mm': m2mm(z1),
        'z2, mm': m2mm(z2),
        'dz, mm': m2mm(dz),
    }

    if len(save_path) > 260:
        raise ValueError(f'Длина имени файла превышает допустимую: len(save_path) = {len(save_path)}; допустимая - 260')

    with open(save_path, 'w') as file:
        for k, v in metadata.items():
            file.write(f'{k}: {v}\n')
