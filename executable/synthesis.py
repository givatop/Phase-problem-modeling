"""
Синтез комплексной амплитуды поля с заданными интенсивностью и фазой.
Сохранение npy-массива и текстового файла метаданных.
"""

import os
import platform
from datetime import date

import numpy as np

from src.propagation.presenter.loader import load_image
from src.propagation.utils.math.general import calculate_chord
import src.propagation.utils.math.units as units
import src.propagation.utils.optic as optic


IS_INTENSITY_FROM_IMAGE = False
IS_PHASE_FROM_IMAGE = False
ADD_NOISE = False
ADD_APERTURE = False
ADD_TILT = False
i_path = None
p_path = None
metadata = {}
folder = '/Users/m.konoplyov/Programming/Python/Phase-problem-modeling/data/executable_synthesis' if platform.system() == 'Darwin' \
        else r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. Проверка корректности FFT1d-решения'

filename = 'i=0.5 phi=sphere 2D complex_field.npy'

# Grid Params
width, height = 512, 512
shape = [width] if height == 1 else [width, height]
x = np.arange(-width // 2, width // 2)
y = np.arange(-height // 2, height // 2)
X, Y = np.meshgrid(x, y)
px_size = units.um2m(5)
wavelength = units.nm2m(555)

# Intensity
if IS_INTENSITY_FROM_IMAGE:
    intensity = load_image(i_path)
else:
    i_amplitude = 0.5
    i_wx, i_wy = width / 10, height / 2
    i_x0, i_y0 = 0, 0
    intensity = i_amplitude * np.ones(shape)
    # intensity = gauss_1d(x, a=i_amplitude, w=i_wx, x0=i_x0)

# Phase
if IS_PHASE_FROM_IMAGE:
    phase = load_image(p_path)
else:
    p_amplitude = 1
    p_wx, p_wy = width / 4, height / 2
    p_x0, p_y0 = 0, 0

    # Полу-окружности/-сферы
    radius = 350_000
    sag = 0.01
    chord = calculate_chord(radius, sag)
    # phase = \
    #     optic.hemisphere(X, Y, sag=sag, r=radius, x0=-chord * 0.5, y0=-chord * 0.75) + \
    #     optic.hemisphere(X, Y, sag=sag, r=radius, x0=-chord * 0.5, y0= chord * 0.75) + \
    #     optic.hemisphere(X, Y, sag=sag, r=radius, x0= width * 0.5 - chord * 0.5, y0=-chord * 0.75) + \
    #     optic.hemisphere(X, Y, sag=sag, r=radius, x0= width * 0.5 - chord * 0.5, y0= chord * 0.75)

    # Линзы
    focus = units.mm2m(100)
    phase = optic.lens_2d(
        units.px2m(X, px_size_m=px_size),
        units.px2m(Y, px_size_m=px_size),
        focus,
        wavelength=wavelength,
    )

# Noise
if ADD_NOISE:
    # params
    mean = 0.0
    standard_deviation = 0.01
    # make some noise
    noise = np.random.normal(mean, standard_deviation, size=intensity.shape)
    intensity += noise

# Aperture
if ADD_APERTURE:
    a_wx, a_wy = width // 2, height // 2
    a_x0, a_y0 = 0, 0
    if intensity.ndim == 1:
        aperture = optic.rect_1d(x, a=1, w=a_wx, x0=a_x0)
    elif intensity.ndim == 2:
        aperture = optic.rect_2d(x, y, a=1, wx=a_wx, wy=a_wy, x0=a_x0, y0=a_y0)
else:
    aperture = np.ones(shape)

# Complex Field
complex_field = np.sqrt(intensity) * np.exp(1j * phase) * aperture

# Tilt
if ADD_TILT:
    distance = units.mm2m(100)
    shift = units.px2m(width // 2, px_size_m=px_size)
    alpha = np.arctan(shift / distance)
    theta = units.degree2rad(45)
    complex_field = optic.add_tilt(X, Y, complex_field, wavelength, alpha, theta)

# Save
if not os.path.exists(folder):
    os.mkdir(folder)
filepath = os.path.join(folder, filename)
np.save(filepath, complex_field)

# region Metadata
filename += '.metadata'
filepath = os.path.join(folder, filename)

metadata['date'] = date.today().strftime("%d/%m/%Y")
metadata['width, px'] = width
metadata['height, px'] = height

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
    metadata['radius, px'] = radius
    metadata['sag, px'] = sag
    metadata['chord, px'] = chord

if ADD_NOISE:
    metadata['noise: mean'] = mean
    metadata['noise: standard_deviation'] = standard_deviation

if ADD_TILT:
    metadata['tilt: distance, m'] = distance
    metadata['tilt: shift, m'] = shift
    metadata['tilt: alpha, rad'] = alpha
    metadata['tilt: theta, rad'] = theta

with open(filepath, 'w') as file:
    for k, v in metadata.items():
        file.write(f'{k}: {v}\n')
# endregion
