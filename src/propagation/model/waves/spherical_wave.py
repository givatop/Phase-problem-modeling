import numpy as np

from ..areas.grid import CartesianGrid
from ...model.areas.aperture import Aperture
from ...model.waves.interface.wave import Wave
from ...utils.math.general import calc_amplitude
from ...utils.math.general import calculate_radius
from ...utils.optic.field import gauss_2d
from ...utils.optic.propagation_methods import angular_spectrum_propagation

from ...utils.math.units import (
    px2m,
    m2mm,
    rad2mm
)


class SphericalWave(Wave):
    """ Волна со сферической аберрацией или сходящаяся сферическая волна """

    def __init__(self, grid: CartesianGrid, focal_len: float, gaussian_width_param: int, wavelength: float):
        """
        Создание распределения поля на двухмерной координатной сетке
        :param grid: двухмерная координатная сетка расчёта распределения поля
        :param focal_len: фокусное расстояние [м]
        :param gaussian_width_param: ширина гауссоиды на уровне интенсивности 1/e^2 [px]
        :param wavelength: длина волны [м]
        """

        self._focal_len = focal_len
        self._gaussian_width_param = gaussian_width_param

        # задание распределения интенсивности волны
        y_grid, x_grid = grid.grid
        gaussian_width_param = px2m(gaussian_width_param, px_size_m=grid.pixel_size)
        intensity = gauss_2d(x_grid, y_grid,
                             wx=gaussian_width_param / 4,
                             wy=gaussian_width_param / 4)

        # волновой вектор
        k = 2 * np.pi / wavelength
        # задание распределения комлексной амплитуды поля
        radius_vector = np.sqrt(x_grid ** 2 + y_grid ** 2 + focal_len ** 2)
        phase = radius_vector * k
        # field = np.sqrt(intensity) * np.exp(-1j * phase)

        super().__init__(intensity, phase, grid, wavelength)

    def get_wavefront_radius(self, aperture: Aperture) -> float:
        """
        Возвращает радиус волнового фронта, найденный по следующей формуле:
        r = (s / 2) + (l ** 2 / (8 * s))
        s - стрелка прогиба
        l - хорда, являющаяся диаметром апертуры
        :param aperture: апертура (circ) для обрезания поля
        :return: радиус волнового фронта при заданной обрезающей апертуре
        """
        # развернутая фаза, обрезанная апертурой
        cut_phase = self.get_unwrapped_phase(aperture=aperture)

        # поиск стрелки прогиба
        amplitude = calc_amplitude(cut_phase)
        sagitta = rad2mm(amplitude, self._wavelength)

        # определение радиуса кривизны волнового фронта
        ap_diameter = m2mm(aperture.aperture_diameter)
        wavefront_radius = calculate_radius(sagitta, ap_diameter)

        return wavefront_radius

    def propagate_on_distance(self, z: float, method=angular_spectrum_propagation, **kwargs):
        method(self, z, **kwargs)

    @property
    def focal_len(self) -> float:
        """
        Фокусное расстояние [м]
        :return:
        """
        return self._focal_len

    @property
    def gaussian_width_param(self) -> float:
        """
        Размер гауссоиды на уровне 1/e^2 в [px]
        :return:
        """
        return self._gaussian_width_param
