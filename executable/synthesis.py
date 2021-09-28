"""
Синтез комплексной амплитуды поля с заданными интенсивностью и фазой.
Сохранение npy-массива и текстового файла метаданных.
"""

import os
import platform
from datetime import date

import numpy as np

from src.propagation.presenter.loader import load_image
from src.propagation.utils.math.units import (nm2m, mm2m, um2m, m2mm, px2m, m2nm, m2um)
from src.propagation.utils.optic import (
    rect_1d,
    rect_2d,
    circ,
    circ_cartesian,
    triangle_1d,
    triangle_2d,
    gauss_1d,
    gauss_2d,
    logistic_1d,
    sin_1d,
    cos_1d,
)

IS_INTENSITY_FROM_IMAGE = False
IS_PHASE_FROM_IMAGE = False
ADD_NOISE = False
i_path = None
p_path = None
metadata = {}
folder = '/Users/megamot/Programming/Python/Phase-problem-modeling/data/executable_synthesis' if platform.system() == 'Darwin' \
        else r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. Проверка корректности FFT1d-решения'

filename = 'i=0.5 phi=sphere 1D complex_field.npy'

# region Grid Params
width, height = 1024, 1
px_size = um2m(5)
wavelength = nm2m(555)
x = np.arange(-width // 2, width // 2)
y = np.arange(-height // 2, height // 2)
X, Y = np.meshgrid(x, y)
# Y, X = np.mgrid[
#         -height // 2: height // 2,
#         -width // 2: width // 2,
#     ]
# endregion

# region Intensity
if IS_INTENSITY_FROM_IMAGE:
    intensity = load_image(i_path)
else:
    i_amplitude = 0.5
    i_wx, i_wy = width / 10, height / 2
    i_x0, i_y0 = 0, 0
    intensity = i_amplitude
    # intensity = gauss_1d(x, a=i_amplitude, w=i_wx, x0=i_x0)
# endregion
# region Phase
if IS_PHASE_FROM_IMAGE:
    phase = load_image(p_path)
else:
    p_amplitude = 1.
    p_wx, p_wy = width / 4, height / 2
    p_x0, p_y0 = p_wx // 4, 0
    focus = mm2m(100)
    phase = np.sqrt(px2m(x, px_size_m=px_size) ** 2 + focus ** 2) * 2 * np.pi / wavelength
    # phase = sin_1d(x, a=p_amplitude, T=p_wx, x0=p_x0)
# endregion

# noise
if ADD_NOISE:
    # params
    mean = 0.0
    standard_deviation = 0.01
    # make some noise
    noise = np.random.normal(mean, standard_deviation, size=intensity.shape)
    intensity += noise
    # write to metadata
    metadata['ADD_NOISE'] = ADD_NOISE
    metadata['mean'] = mean
    metadata['standard_deviation'] = standard_deviation

# region Complex Field
complex_field = np.sqrt(intensity) * np.exp(-1j * phase)
# endregion
# region Save Complex Field
filepath = os.path.join(folder, filename)
np.save(filepath, complex_field)
# endregion
# region Metadata
filename += '.metadata'
filepath = os.path.join(folder, filename)

metadata['date'] = date.today().strftime("%d/%m/%Y")
metadata['width, px'] = width
metadata['height, px'] = height
metadata['wavelength, nm'] = m2nm(wavelength)
metadata['px_size, um'] = m2um(px_size)

if IS_INTENSITY_FROM_IMAGE:
    metadata['i_path'] = i_path
else:
    metadata['i_amplitude, px'] = i_amplitude
    metadata['i_wx, px'] = i_wx
    metadata['i_wy, px'] = i_wy
    metadata['i_x0, px'] = i_x0
    metadata['i_y0, px'] = i_y0

if IS_PHASE_FROM_IMAGE:
    metadata['p_path'] = p_path
else:
    metadata['p_amplitude, px'] = p_amplitude
    metadata['p_wx, px'] = p_wx
    metadata['p_wy, px'] = p_wy
    metadata['p_x0, px'] = p_x0
    metadata['p_y0, px'] = p_y0

with open(filepath, 'w') as file:
    for k, v in metadata.items():
        file.write(f'{k}: {v}\n')
# endregion
