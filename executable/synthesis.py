"""
Синтез комплексной амплитуды поля с заданными интенсивностью и фазой.
Сохранение npy-массива и текстового файла метаданных.
"""

import os
import numpy as np

from src.propagation.presenter.loader import load_image
from src.propagation.utils.math.units import (nm2m, mm2m, um2m, m2mm, px2m)
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
i_path = None
p_path = None
metadata = {}
folder = r'C:\Users\IGritsenko\Documents'
filename = 'test_complex_field.npy'


# region Grid Params
width, height = 1024, 1024
Y, X = np.mgrid[
        -height // 2: height // 2,
        -width // 2: width // 2,
    ]
# endregion

# region Intensity
if IS_INTENSITY_FROM_IMAGE:
    intensity = load_image(i_path)
else:
    i_amplitude = 0.5
    i_wx, i_wy = width/2, height/2
    i_x0, i_y0 = 0, 0
    intensity = i_amplitude
# endregion
# region Phase
if IS_PHASE_FROM_IMAGE:
    phase = load_image(p_path)
else:
    p_amplitude = 1.
    p_wx, p_wy = width/2, height/2
    p_x0, p_y0 = 0, 0
    phase = gauss_2d(X, Y, wx=p_wx, wy=p_wy, a=p_amplitude)
# endregion

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

metadata['width, пкс'] = width
metadata['height, пкс'] = height

if IS_INTENSITY_FROM_IMAGE:
    metadata['i_path'] = i_path
else:
    metadata['i_amplitude, пкс'] = i_amplitude
    metadata['i_wx, пкс'] = i_wx
    metadata['i_wy, пкс'] = i_wy
    metadata['i_x0, пкс'] = i_x0
    metadata['i_y0, пкс'] = i_y0

if IS_PHASE_FROM_IMAGE:
    metadata['p_path'] = p_path
else:
    metadata['p_amplitude, пкс'] = p_amplitude
    metadata['p_wx, пкс'] = p_wx
    metadata['p_wy, пкс'] = p_wy
    metadata['p_x0, пкс'] = p_x0
    metadata['p_y0, пкс'] = p_y0

with open(filepath, 'w') as file:
    for k, v in metadata.items():
        file.write(f'{k}: {v}\n')
# endregion
