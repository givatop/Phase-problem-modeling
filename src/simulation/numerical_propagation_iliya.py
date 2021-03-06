from icecream import ic

from src.propagation.model.areas.aperture import Aperture
from src.propagation.model.areas.grid import PolarGrid
from src.propagation.model.areas.grid import CartesianGrid
from src.propagation.model.waves.spherical_wave import SphericalWave
from src.propagation.utils.math.units import (nm2m, mm2m, um2m, m2mm, percent2decimal)
from src.propagation.utils.math.general import *
from src.propagation.utils.optic.propagation_methods import angular_spectrum_bl_propagation
from src.propagation.presenter.presenter_interface import (
    save_intensity_plot,
    save_phase_plot,
    save_phase_npy,
    save_intensity_npy,
    save_r_z_metadata,
)

# основные параметры для синтеза волны
width, height = 1024, 1024
wavelength = nm2m(632.8)
px_size = um2m(5.04)
gaussian_width_params = [250]
focal_lens = [100]
focal_lens = list(map(mm2m, focal_lens))

# вариации порога определения апертуры
thresholds = [np.exp(-2), percent2decimal(13), percent2decimal(0.5), percent2decimal(0.8)]
t_num = 0

# параметры для итерации при рапространении волны
start = mm2m(0)
stop = mm2m(100)
step = mm2m(25)
distances = np.arange(start, stop + step, step)

# матрица в квадратичных координатах
square_area_1 = CartesianGrid(height, width, pixel_size=px_size)
radial_area_1 = PolarGrid(square_area_1)

for focal_len in focal_lens:
    for gaussian_width_param in gaussian_width_params:

        # конфигурация
        folder_name = \
            f'z_{m2mm(start)}-{m2mm(stop)}-{m2mm(step)} ' \
            f'f_{m2mm(focal_len)} ' \
            f'w_{gaussian_width_param} ' \
            f'{width}x{height}'

        for z in distances:
            # создание сферической волны
            field = SphericalWave(square_area_1, focal_len, gaussian_width_param, wavelength)

            # распространение волны на дистанцию z
            field.propagate_on_distance(z, method=angular_spectrum_bl_propagation)
            # todo мы теряем U(z=0) и на каждой итерации цикла приходится генерить её заново

            # определение апертуры для поиска радиуса волнового фронта
            aperture = Aperture(radial_area_1, field, z, thresholds[t_num])

            # радиус волнового фронта просто для вывода
            r = field.get_wavefront_radius(aperture=aperture)
            ic(m2mm(z), r)

            # построение графиков для снапшотов
            save_r_z_metadata(folder_name, 'r(z).txt', r, m2mm(z))
            # WavePlotter.save_phase(field, aperture, z, saver, save_npy=True)
            # WavePlotter.save_intensity(field, z, saver, save_npy=True)

        ic()
