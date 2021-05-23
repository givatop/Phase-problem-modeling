from icecream import ic

from src.propagation.model.areas.aperture import Aperture
from src.propagation.model.areas.grid import RadialCoordinateGrid
from src.propagation.model.areas.grid import CoordinateGrid
from src.propagation.presenter.interface.wave_plotter import WavePlotter
from src.propagation.presenter.saver.simple_saver import SimpleSaver
from src.propagation.model.waves.spherical_wave import SphericalWave
from src.propagation.utils.math import units
from src.propagation.utils.math.general import *
from src.propagation.utils.optic.propagation_methods import angular_spectrum_bl_propagation

# основные параметры для синтеза волны
width, height = 512, 512
wavelength = units.nm2m(632.8)
px_size = units.um2m(5.04)
gaussian_width_params = [250]
focal_lens = [100]
focal_lens = list(map(units.mm2m, focal_lens))

# вариации порога определения апертуры
thresholds = [np.exp(-2), units.percent2decimal(13), units.percent2decimal(0.5), units.percent2decimal(0.8)]
t_num = 0

# параметры для итерации при рапространении волны
start = units.mm2m(0)
stop = units.mm2m(200)
step = units.mm2m(25)
distances = np.arange(start, stop + step, step)

# матрица в квадратичных координатах
square_area_1 = CoordinateGrid(height, width, pixel_size=px_size)
radial_area_1 = RadialCoordinateGrid(square_area_1)

for focal_len in focal_lens:
    for gaussian_width_param in gaussian_width_params:

        # конфигурация
        folder_name = \
            f'z_{units.m2mm(start)}-{units.m2mm(stop)}-{units.m2mm(step)} ' \
            f'f_{units.m2mm(focal_len)} ' \
            f'w_{gaussian_width_param} ' \
            f'{width}x{height}'
        saver = SimpleSaver(folder_name)

        for z in distances:
            # создание сферической волны
            field = SphericalWave(square_area_1, focal_len, gaussian_width_param, wavelength, z)

            # распространение волны на дистанцию z
            field.propagate_on_distance(z, method=angular_spectrum_bl_propagation)
            # todo мы теряем U(z=0) и на каждой итерации цикла приходится генерить её заново

            # определение апертуры для поиска радиуса волнового фронта
            aperture = Aperture(radial_area_1, widest_diameter(field.intensity, thresholds[t_num]))

            # радиус волнового фронта просто для вывода
            r = field.get_wavefront_radius(aperture)
            ic(z, r)

            # построение графиков для снапшотов
            # WavePlotter.write_r_z(r, z, saver)
            # WavePlotter.save_phase(field, aperture, z, saver, save_npy=True)
            WavePlotter.save_intensity(field, z, saver, save_npy=True)

        ic()
