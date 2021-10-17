from icecream import ic

from src.propagation.model.areas.aperture import Aperture
from src.propagation.model.areas.grid import CartesianGrid, FrequencyGrid
from src.propagation.model.areas.grid import PolarGrid
from src.propagation.model.waves.spherical_wave import SphericalWave
from src.propagation.presenter.presenter_interface import (
    save_intensity_plot,
    save_phase_plot,
    save_phase_npy,
    save_intensity_npy,
    save_r_z_metadata,
)
from src.propagation.presenter.saver import create_folder_name, create_filename
from src.propagation.utils.math import units
from src.propagation.utils.math.general import *

# основные параметры для синтеза волны
from src.propagation.utils.optic.propagation_methods import angular_spectrum_bl_propagation

width, height = 1024, 1024
wavelength = units.nm2m(632.8)
px_size = units.um2m(5.04)
gaussian_width_params = [200]
focal_lens = [100]
focal_lens = list(map(units.mm2m, focal_lens))

# вариации порога определения апертуры
thresholds = [np.exp(-2), units.percent2decimal(13), units.percent2decimal(0.5), units.percent2decimal(0.8)]
t_num = 0

# параметры для итерации при рапространении волны
start = units.mm2m(85)
stop = units.mm2m(200)
step = units.mm2m(1)
distances = np.arange(start, stop + step, step)

# матрица в квадратичных координатах
square_area_1 = CartesianGrid(height, width, pixel_size=px_size)
radial_area_1 = PolarGrid(square_area_1)
freq_grid = FrequencyGrid(square_area_1)

for focal_len in focal_lens:
    for gaussian_width_param in gaussian_width_params:

        field = SphericalWave(square_area_1, focal_len, gaussian_width_param, wavelength)
        folder_name = create_folder_name(start, stop, step, field)

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
            ic(z, r)

            # построение графиков для снапшотов
            filename = create_filename(z, extension='png')

            save_intensity_plot(
                folder_name,
                filename,
                field.intensity
            )

            save_phase_plot(
                folder_name,
                filename,
                field.get_wrapped_phase(aperture),
                field.get_unwrapped_phase(aperture),
                wavelength,
                r,
                z
            )

            filename = create_filename(z, extension='npy')
            save_phase_npy(folder_name, filename, field.phase)
            save_intensity_npy(folder_name, filename, field.intensity)

            save_r_z_metadata(
                folder_name=folder_name,
                filename='R (z)',
                r=r,
                z=z
            )

        ic()
