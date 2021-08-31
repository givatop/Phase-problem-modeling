import os
from datetime import datetime

import numpy as np
from tools.optic.field import circ

DATA_NOW = datetime.now().strftime('%d.%m.%y')
TIME_NOW = datetime.now().strftime('%H_%M_%S')
OUTPUT_DIR = fr'lens_propagation {DATA_NOW}'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

show_lens_plot = 1
show_initfield_phase = 1
show_initfield_intensity = 1
show_zfield = 1


def lens_tf(X, Y, focal_length, radius, wavelength) -> np.ndarray:
    """
    lens Transmittance Function - функция пропускания тонкой линзы
    :param X: координаты зрачка в [m]
    :param Y: координаты зрачка в [m]
    :param focal_length: фокусное расстояние в [m]
    :param radius: световой радиус в [m]
    :param wavelength: длина волны в [m]
    :return:
    """
    aperture = circ(np.sqrt(X**2 + Y**2), w=2*radius).astype(np.complex128)
    phase_coef = np.exp(-1j * np.pi * (X**2 + Y**2) / (wavelength * focal_length))  # todo откуда-то берутся круги!
    return aperture * phase_coef