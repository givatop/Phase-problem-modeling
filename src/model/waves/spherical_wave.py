from icecream import ic
from numpy.fft import fft2, fftshift, ifft2
from skimage.restoration import unwrap_phase

from src.utils.math import units
from src.model.waves.interface.wave import Wave
from src.model.areas.interface.aperture import Aperture
from src.model.areas.interface.area import Area
from src.utils.optic.field import *
from src.utils.math.general import *


# класс волны со сферической аберрацией или сходящейся сферической волны
class SphericalWave(Wave):

    def __init__(self, ar: Area, focal_len: float, gaussian_width_param: int, wavelength: float):
        """
        Создание распределения поля на двухмерной координатной сетке
        :param ar: двухмерная координатная сетка расчёта распределения поля
        :param focal_len: фокусное расстояние [м]
        :param gaussian_width_param: ширина гауссоиды на уровне интенсивности 1/e^2 [px]
        :param wavelength: длина волны [м]
        """
        y_grid_array, x_grid_array = ar.coordinate_grid
        radius_vector = np.sqrt(x_grid_array ** 2 + y_grid_array ** 2 + focal_len ** 2)
        # волновой вектор
        k = 2 * np.pi / wavelength

        self.__gaussian_width_param = gaussian_width_param
        gaussian_width_param = units.px2m(gaussian_width_param, px_size_m=ar.pixel_size)

        self.__intensity = gauss_2d(x_grid_array, y_grid_array, wx=gaussian_width_param / 4,
                                    wy=gaussian_width_param / 4)
        self.__field = np.sqrt(self.__intensity) * np.exp(-1j * k * radius_vector)
        self.__phase = np.angle(self.__field)
        self.__wavelength = wavelength
        self.__focal_len = focal_len
        self.__area = ar

    def get_wrapped_phase(self, aperture=None) -> np.ndarray:
        if aperture:
            return self.__phase * aperture.aperture
        else:
            return self.__phase

    def get_unwrapped_phase(self, aperture=None) -> np.ndarray:
        if aperture:
            return unwrap_phase(self.__phase * aperture.aperture)
        else:
            return unwrap_phase(self.__phase)

    def get_wavefront_radius(self, aperture: Aperture) -> float:
        # развернутая фаза, обрезанная апертурой
        cut_phase = self.get_unwrapped_phase(aperture=aperture)

        mask2 = cut_phase == 0
        cut_phase[mask2] = np.max(cut_phase)
        cut_phase -= cut_phase.min()

        cut_phase[mask2] = 0

        # поиск стрелки прогиба
        a = calc_amplitude(cut_phase)
        saggita = units.rad2mm(a, self.__wavelength)

        # определение радиуса кривизны волнового фронта
        ap = units.m2mm(aperture.aperture_diameter)
        wavefront_radius = calculate_radius(saggita, ap)
        ic(aperture.aperture_diameter)
        ic(saggita)

        return wavefront_radius

    def propagate_on_distance(self, z: float, method='angular_spectrum'):
        if method == 'angular_spectrum':
            self.__angular_spectrum_propagation(z)

    def __angular_spectrum_propagation(self, z: float):
        """
        Метод распространения (преобразования) волны методом углового спектра
        :param z: дистанция распространения
        :return:
        """

        height = self.__field.shape[0]  # количество строк матрицы
        width = self.__field.shape[1]  # количество элеметов в каждой строке матрицы

        # волновое число
        wave_number = 2 * np.pi / self.__wavelength

        # создание сетки в частотной области при условии выполнения теоремы Котельникова
        nu_x = np.arange(-width / 2, width / 2) / (width * self.__area.pixel_size)
        nu_y = np.arange(-height / 2, height / 2) / (height * self.__area.pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)

        # сдвиг высоких частот к краям сетки
        nu_x_grid, nu_y_grid = fftshift(nu_x_grid), fftshift(nu_y_grid)

        # Фурье-образ исходного поля
        field = fft2(self.__field)

        # передаточная функция слоя пространства
        exp_term = np.sqrt(
            1 - (self.__wavelength * nu_x_grid) ** 2 -
            (self.__wavelength * nu_y_grid) ** 2)
        h = np.exp(1j * wave_number * z * exp_term)

        # обратное преобразование Фурье
        self.__field = ifft2(field * h)

        self.__phase = np.angle(self.__field)
        self.__intensity = np.abs(self.__field) ** 2

    @property
    def field(self) -> np.ndarray:
        return self.__field

    @property
    def area(self) -> Area:
        return self.__area

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.__field)

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.__field) ** 2

    @property
    def wavelength(self) -> float:
        return self.__wavelength

    @property
    def focal_len(self) -> float:
        return self.__focal_len

    @property
    def gaussian_width_param(self) -> float:
        return self.__gaussian_width_param

    @field.setter
    def field(self, field):
        self.__field = field

    @area.setter
    def area(self, area):
        self.__area = area

    @wavelength.setter
    def wavelength(self, wavelength):
        self.__wavelength = wavelength

    @gaussian_width_param.setter
    def gaussian_width_param(self, gaussian_width_param):
        self.__gaussian_width_param = gaussian_width_param

    @focal_len.setter
    def focal_len(self, focal_len):
        self.__focal_len = focal_len

    @intensity.setter
    def intensity(self, intensity):
        self.__intensity = intensity

    @phase.setter
    def phase(self, phase):
        self.__phase = phase
