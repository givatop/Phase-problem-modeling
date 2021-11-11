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
        # Ограничение фазы апертурой
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

        print(f'l: {units.m2mm(chord)} mm')
        print(f's: {units.m2rad(sag, wavelength)} rad')
        print(f'r: {units.m2mm(radius)} mm')
        return radius

    else:
        raise ValueError(f"{intensity.ndim} dimensions is not supported")


if __name__ == '__main__':
    import src.propagation.utils.math.units as units

    wave_len = 555e-9
    s = units.rad2m(209.77, wave_len)
    l = 0.001925 * 2  # m
    r = calculate_radius(s, l)  # m

    print(f'{units.m2mm(r)} mm')  # 99.9987907571921 mm
