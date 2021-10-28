import numpy as np

import src.propagation.utils.math.units as units
import src.propagation.utils.optic as optic
from src.propagation.model.areas.aperture import widest_diameter
from src.propagation.utils.math.general import calculate_radius, calc_amplitude


def find_radius(intensity: np.ndarray, phase: np.ndarray, wavelength, px_size, threshold=np.exp(-2)):
    """
    Поиск радиуса кривизны сферической волны
    :param intensity:
    :param phase:
    :param wavelength: [м]
    :param px_size: [м]
    :param threshold:
    :return: [м]
    """

    if intensity.ndim == 1:
        width = intensity.shape[0]
        # Координатная сетка
        x = units.px(np.arange(-width / 2, width / 2))
        x = units.px2m(x, px_size)
        # Хорда
        chord = units.px(widest_diameter(intensity, threshold, axis=0))
        chord = units.px2m(chord, px_size)
        # Апертура
        aperture = optic.rect_1d(x, a=1, w=chord, x0=0, y0=0)
        nonzero_indices = aperture.nonzero()
        first_nonzero_value = phase[nonzero_indices[0][0]]
        last_nonzero_value = phase[nonzero_indices[0][-1]]
        nonzero_value = first_nonzero_value if abs(first_nonzero_value > last_nonzero_value) else last_nonzero_value
        limited_phase = (phase - nonzero_value) * aperture
        # Стрелка прогиба
        sag = units.radians(calc_amplitude(limited_phase))
        sag = units.rad2m(sag, wavelength)
        # Радиус
        radius = calculate_radius(sag, chord)
        return radius

    else:
        raise ValueError(f"{intensity.ndim} dimensions is not supported")


if __name__ == '__main__':
    import os
    import numpy as np
    from icecream import ic
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from src.propagation.presenter.loader import load_image, load_files

    folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\2. Экспериментальные\4. cs2100 21.10.2021\3'
    filename_1 = 'yslice z=0.000.npy'
    filename_2 = 'TIE yslice z=0.000 yslice z=1.000 dz=1.000mm.npy'
    filepath_1 = os.path.join(folder, filename_1)
    filepath_2 = os.path.join(folder, filename_2)
    base_filename_1 = os.path.splitext(os.path.basename(filename_1))[0]
    base_filename_2 = os.path.splitext(os.path.basename(filename_2))[0]
    wavelength = 532e-9
    px_size = 5.06e-6

    intensity = np.load(filepath_1)
    phase = np.load(filepath_2)

    radius = find_radius(intensity, phase, wavelength, px_size)

    ic(radius)
