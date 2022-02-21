import os
import numpy as np

import src.propagation.utils.math.units as units
import src.propagation.utils.optic as optic
from src.propagation.presenter.loader import load_file


folder = r"\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа" + \
         r"\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\1. Изображения"
filename = 'BMSTU -b 1024x1024.png'
path = os.path.join(folder, filename)

phase = load_file(path)
intensity = np.ones(phase.shape)

height, width = intensity.shape
x = np.arange(-width / 2, width / 2)
y = np.arange(-height / 2, height / 2)
X, Y = np.meshgrid(x, y)

aperture = optic.rect_2d(X, Y, wx=0.9 * width, wy=0.9 * height)
phase, intensity = phase * aperture, intensity * aperture

field = np.sqrt(intensity) * np.exp(1j * phase) * aperture

filename = f'z=0.000.npy'
path = os.path.join(folder, filename)
np.save(path, field)
