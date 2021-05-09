from icecream import ic

from src.propagation.model.areas.radial_aperture import RadialAperture
from src.propagation.model.areas.radial_area import RadialArea
from src.propagation.model.areas.square_area import SquareArea
from src.propagation.presenter.interface.wave_plotter import WavePlotter
from src.propagation.presenter.saver.mac_saver import MacSaver
from src.propagation.model.waves.spherical_wave import SphericalWave
from src.propagation.utils.math import units
from src.propagation.utils.math.general import *
from src.propagation.utils.optic.propagation_methods import angular_spectrum_bl_propagation

# конфигурация
saver = MacSaver()

# основные параметры для синтеза волны
wavelength = 659.6e-9
px_size = 5.04e-6
focal_len = 100e-3
gaussian_width_param = 250

# вариации порога определения апертуры
thresholds = [np.exp(-2), units.percent2decimal(13), units.percent2decimal(0.5), units.percent2decimal(0.8)]
t_num = 0

# параметры для итерации при рапространении волны
start = units.mm2m(0)
stop = units.mm2m(600)
step = units.mm2m(50)
z_array = np.array(np.arange(units.m2mm(start), units.m2mm(stop + step), units.m2mm(step)))

# изменяющийся параметр для выборок
matrixes = np.array([512])

# массивы для записи значений циклов нескольких прогонок
array_wave_array = []
array_aperture_array = []

for matrix in matrixes:
    ic(matrix)
    matrix_size = matrix

    # матрица в квадратичных координатах
    square_area_1 = SquareArea(matrix_size, matrix_size, pixel_size=px_size)

    # массивы для снапшотов
    wave_array = []
    aperture_array = []

    # массивы для одной прогонки
    z_distances_array = []
    wavefront_radius_array = []

    for z in np.arange(start, stop + step, step):
        # синтез фиктивной апертуры
        radial_area_1 = RadialArea(square_area_1)
        aperture = RadialAperture(radial_area_1, 2 * gaussian_width_param)

        # создание сферической волны
        field = SphericalWave(square_area_1, focal_len, gaussian_width_param, wavelength, z)
        field.field *= aperture.aperture

        # распространение волны на дистанцию z
        field.propagate_on_distance(z, method=angular_spectrum_bl_propagation)

        # определение апертуры для поиска радиуса волнового фронта
        aperture = RadialAperture(radial_area_1, widest_diameter(field.intensity, thresholds[t_num]))

        # радиус волнового фронта просто для вывода
        r = field.get_wavefront_radius(aperture)
        ic(r)

        # построение графиков для снапшотов
        WavePlotter.save_phase(field, aperture, z, saver)
        WavePlotter.save_intensity(field, z, saver)

        wave_array.append(field)
        aperture_array.append(aperture)

    ic()
    array_wave_array.append(wave_array)
    array_aperture_array.append(aperture_array)

# построение графиков для нескольких прогонок
WavePlotter.save_r_z(array_wave_array, array_aperture_array, z_array, matrixes, step, saver)